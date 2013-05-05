#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <assert.h>

#include "mpi.h"

#define MAX_CHAR_PER_LINE 128

float ** file_read(char* filename, int *numObjs, int *numCoords)
{
    float ** objects;
    int i;
    int j;
    int len;
    ssize_t numBytesRead;
    
    FILE * infile;
    char * line;
    char * ret;

    int lineLen;

    if((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error: no such file (%s)\n", filename);
        return NULL;
    }
    lineLen = MAX_CHAR_PER_LINE;
    line = (char*) malloc(lineLen);
    assert(line != NULL);

    (*numObjs) = 0;
    
    // read the data file
    while (fgets(line, lineLen, infile) != NULL)
    {
        // check each line to find the max line length
        while (strlen(line) == lineLen-1) {
            // this line read is not complete 
            len = strlen(line);
            fseek(infile, -len, SEEK_CUR);
            lineLen += MAX_CHAR_PER_LINE;
            line = (char*) realloc(line, lineLen);
            assert(line != NULL);
            ret = fgets(line, lineLen, infile);
            assert(ret != NULL);
        }
        if (strtok(line, "\n") != 0)
        {
            (*numObjs)++;
        }
    }

    rewind(infile);

    (*numCoords) = 0;
    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, "\n") != 0) {
            while (strtok(NULL, "\t") != NULL) (*numCoords)++;
            break;
        }
    }

    rewind(infile);

    len = (*numObjs) * (*numCoords);
    objects    = (float**)malloc((*numObjs) * sizeof(float*));
    assert(objects != NULL);
    objects[0] = (float*) malloc(len * sizeof(float));
    assert(objects[0] != NULL);

    for (i=1; i<(*numObjs); i++)
        objects[i] = objects[i-1] + (*numCoords);
    i = 0;

    while (fgets(line, lineLen, infile) != NULL) {
        if (strtok(line, "\n") == NULL) continue;
        for (j=0; j<(*numCoords); j++)
            objects[i][j] = atof(strtok(NULL, "\t"));
    }

    fclose(infile);
    free(line);
        
    return objects;
}

float ** mpi_read(char* filename, 
        int *numObjs,
        int *numCoords,
        MPI_Comm comm)
{
    float ** objects;
    int i,j,len,divd,rem;
    int rank, nproc;
    MPI_Status status;

    MPI_Comm_rank(comm, &rank);    
    MPI_Comm_size(comm, &nproc);

    // only master read the data
    if (rank == 0)
    {
        // if objects are too large, we can read it all
        // into single node, need to improve it later
        objects = file_read(filename, numObjs, numCoords);
        if (NULL == objects)
        {
            *numObjs = -1; // errors
        }
    }

    // broadcast 
    MPI_Bcast(numObjs,   1, MPI_INT, 0, comm);
    MPI_Bcast(numCoords, 1, MPI_INT, 0, comm);

    // check the numObjs
    if (*numObjs == -1)
    {
        // oops!
        MPI_Finalize();
        exit(1);
    }

    divd = (*numObjs / nproc);
    rem = (*numObjs % nproc);

    if (rank == 0)
    {
        // distributes the data
        int index = (rem > 0) ? divd + 1 : divd;
        (*numObjs) = index;
        for(i=1; i<nproc; ++i)
        {
            int msg_size = (i < rem) ? (divd+1) : divd;
            // send it
            MPI_Send(objects[index], msg_size*(*numCoords), MPI_FLOAT,i, i, comm);
            index += msg_size;
        }

        // reduce the objects[] to local size
        objects[0] = realloc(objects[0],(*numObjs)*(*numCoords)*sizeof(float));
        assert(objects[0] != NULL);
        objects = realloc(objects, (*numObjs)*sizeof(float*));
        assert(objects != NULL);
    }
    else
    {
        (*numObjs) = (rank < rem) ? divd+1 : divd;
        objects    = (float**)malloc((*numObjs)*sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*) malloc((*numObjs)*(*numCoords)*sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<(*numObjs); i++)            
        {
            objects[i] = objects[i-1] + (*numCoords);
        }
        // recv data from master
        MPI_Recv(objects[0], (*numObjs)*(*numCoords), MPI_FLOAT, 0, rank, comm, &status);
    }

    return objects;
}

inline static float euclid_dist_2(int numdims, float* coord1, float* coord2)
{
    int i;
    float ans = 0.f;
    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return ans;
}

inline static int find_nearest_cluster(int numClusters, int numCoords, float* object, float** clusters)
{
    int   index, i;
    float dist, min_dist;

    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);
    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        if (dist < min_dist) {
            min_dist = dist;
            index = i;
        }
    }

    return index;
}

int mpi_kmeans(float** objects, int numCoords, int numObjs, int numClusters, int threshold,
        int *membership, float **clusters, MPI_Comm   comm) 
{
    int      i, j, rank, index, loop=0, total_numObjs;
    int     *newClusterSize;
    int     *clusterSize;
    float delta;
    float    delta_tmp;
    float  **newClusters; 

    for (i=0; i<numObjs; i++) membership[i] = -1;
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);
    clusterSize    = (int*) calloc(numClusters, sizeof(int));
    assert(clusterSize != NULL);

    newClusters    = (float**) malloc(numClusters *sizeof(float*));
    assert(newClusters != NULL);
    newClusters[0] = (float*)  calloc(numClusters * numCoords, sizeof(float));
    assert(newClusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        newClusters[i] = newClusters[i-1] + numCoords;
    MPI_Allreduce(&numObjs, &total_numObjs, 1, MPI_INT, MPI_SUM, comm);

    do
    {
        delta = 0.0;
        for (i=0; i<numObjs; i++) {
            index = find_nearest_cluster(numClusters, numCoords, objects[i],  clusters);
            if (membership[i] != index) delta += 1.0;
            membership[i] = index;
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index][j] += objects[i][j];
        }

        MPI_Allreduce(newClusters[0], clusters[0], numClusters*numCoords, MPI_FLOAT, MPI_SUM, comm);
        MPI_Allreduce(newClusterSize, clusterSize, numClusters, MPI_INT, MPI_SUM, comm);

        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (clusterSize[i] > 1)
                    clusters[i][j] /= clusterSize[i];
                newClusters[i][j] = 0.0;
            }

            newClusterSize[i] = 0;
        }

        MPI_Allreduce(&delta, &delta_tmp, 1, MPI_FLOAT, MPI_SUM, comm);
        delta = delta_tmp / total_numObjs;
    } while(delta > threshold && loop++ < 500);

    free(newClusters[0]);    
    free(newClusters);    
    free(newClusterSize);    
    free(clusterSize);

    return 0;

}

int mpi_write(char* filename, int numClusters, int numObjs, int numCoords,
        float **clusters, int *membership,  int totalNumObjs, MPI_Comm   comm)
{
    int        divd, rem, len, err;
    int        i, j, k, rank, nproc;
    char       outFileName[1024], fs_type[32], str[32], *delim;
    MPI_File   fh;
    MPI_Status status;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);

    delim = filename;
    if (rank == 0)
    {
        printf("Writing coordinates of K=%d cluster centers to file \"%s.cluster_centres\"\n", numClusters, delim);
        sprintf(outFileName, "%s.cluster_centres", filename);
        err = MPI_File_open(MPI_COMM_SELF, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) {
            char errstr[MPI_MAX_ERROR_STRING];
            int  errlen;
            MPI_Error_string(err, errstr, &errlen);
            printf("Error at opening file %s (%s)\n", outFileName,errstr);
            MPI_Finalize();
            exit(1);
        }

        for (i=0; i<numClusters; i++) {
            sprintf(str, "%d ", i);
            MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
            for (j=0; j<numCoords; j++) {
                sprintf(str, "%f ", clusters[i][j]);
                MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
            }
            MPI_File_write(fh, "\n", 1, MPI_CHAR, &status);
        }
        MPI_File_close(&fh);


        printf("Writing membership of N=%d data objects to file \"%s.membership\"\n", totalNumObjs, delim);

        int divd = totalNumObjs / nproc;
        int rem  = totalNumObjs % nproc;
        sprintf(outFileName, "%s.membership", filename);
        err = MPI_File_open(MPI_COMM_SELF, outFileName, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        if (err != MPI_SUCCESS) {
            char errstr[MPI_MAX_ERROR_STRING];
            int  errlen;
            MPI_Error_string(err, errstr, &errlen);
            printf("Error at opening file %s (%s)\n", outFileName,errstr);
            MPI_Finalize();
            exit(1);
        }

        for (j=0; j<numObjs; j++) {
            sprintf(str, "%d %d\n", j, membership[j]);
            MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
        }

        k = numObjs;

        for (i=1; i<nproc; i++) {
            numObjs = (i < rem) ? divd+1 : divd;
            MPI_Recv(membership, numObjs, MPI_INT, i, i, comm, &status);
            for (j=0; j<numObjs; j++) {
                sprintf(str, "%d %d\n", k++, membership[j]);
                MPI_File_write(fh, str, strlen(str), MPI_CHAR, &status);
            }
        }

        MPI_File_close(&fh);
    }
    else
    {
        MPI_Send(membership, numObjs, MPI_INT, 0, rank, comm);
    }


    return 0;

}




int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage:%s data_file num_clusters\n", argv[0]);
        return -1;
    }

    int i;
    int j;

    int numClusters; // number of cluster
    int numCoords;   // number of features
    int numObjs; // number of instances, each node contains (numObjs) objects
    int totalNumObjs;

    int *membership; // [numObjs]: object belongs to which cluster center

    float **objects; // [numObjs][numCoords]
    float **clusters; // [numClusters][numCoords]
    float threshold;


    int rank;
    int nproc;

    char* filename;
    int mpi_namelen;
    char mpi_name[mpi_namelen];

    MPI_Status status;

    
    // the common steps
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Get_processor_name(mpi_name,&mpi_namelen);

    // set default values
    threshold        = 0.001;    
    numClusters      = 0;
    filename         = NULL;


    filename = argv[1];
    numClusters = atoi(argv[2]);

    if (filename == NULL || numClusters <= 1)
    {
        fprintf(stderr, "File name can not be empty or clusters number should >= 2\n");
        if (rank == 0)
        {
            MPI_Finalize();
            exit(-1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    objects = mpi_read(filename, &numObjs, &numCoords, MPI_COMM_WORLD);
    clusters    = (float**) malloc(numClusters * sizeof(float*));
    assert(clusters != NULL);
    clusters[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
    assert(clusters[0] != NULL);
    for (i=1; i<numClusters; i++)
        clusters[i] = clusters[i-1] + numCoords;

    MPI_Allreduce(&numObjs, &totalNumObjs, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // pick first numClusters elements as initial cluster centers
    if (rank == 0)
    {
        for (i=0; i<numClusters; i++)
            for (j=0; j<numCoords; j++)
                clusters[i][j] = objects[i][j];
    }

    // broadcast all clusters to other nodes
    MPI_Bcast(clusters[0], numClusters*numCoords, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // allocate the memory for the membership
    membership = (int*) malloc(numObjs * sizeof(int));
    assert(membership != NULL);

    // do kmean clustering 
    mpi_kmeans(objects, numCoords, numObjs, numClusters, threshold, membership, clusters, MPI_COMM_WORLD);
    

    // free memory
    free(objects[0]);
    free(objects);

    // write the results into file
    mpi_write(filename, numClusters, numObjs, numCoords,clusters, membership, totalNumObjs, MPI_COMM_WORLD);
    free(membership);
    free(clusters[0]);
    free(clusters);


    // done
    MPI_Finalize();

    return 0;
}
