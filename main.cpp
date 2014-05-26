#include <iostream>
#include <assert.h>
using namespace std;

#include "kmeans.h"

int main()
{
    double **objects;
    const int cluster_num = 4;
    int rowNum = 10;
    int colObjNum = 8;
    const char *filename = "D:/DataForHIT/CB/DM_KMeans/color100.txt";

    KMeans* kmeans = new KMeans(cluster_num, rowNum, colObjNum);
    objects = kmeans->file_read(filename, &rowNum, &colObjNum);

    int *Labels = new int[rowNum];
    assert(Labels != NULL);

    kmeans->Cluster(objects, rowNum, colObjNum, Labels);
    if (_DEBUG == true)
        kmeans->OutCLusterMeans(Labels);
    kmeans->file_write(filename, Labels);
    delete[] Labels;
    Labels = NULL;
    for(int i = 0; i < rowNum; i ++) {
        delete[] objects[i];
        objects[i] = NULL;
    }
    delete[] objects;
    objects = NULL;
    delete kmeans;
    kmeans = NULL;
    return 0;
}
