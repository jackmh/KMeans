#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <assert.h>
#include <math.h>
using namespace std;

#include "kmeans.h"

KMeans::KMeans(int clusterNum, int rowNum, int colNum)
{
    m_clusterNum = clusterNum;
    m_rowNum = rowNum;
    m_colNum = colNum;

    m_Means = new double*[m_clusterNum];
    if (m_Means == NULL) exit(-1);

    m_maxIterNum = 1000;
    m_endError = 0.001;
}

KMeans::~KMeans()
{
    for (int i = 0; i < m_clusterNum; ++ i)
    {
        delete[] m_Means[i];
        m_Means[i] = NULL;
    }
    delete[] m_Means;
    m_Means = NULL;
}

double** KMeans::file_read(const char *filename, int  *rowNum, int  *colObjNum)
{
    FILE *infile;
    char *line;
    if ((infile = fopen(filename, "r")) == NULL)
    {
        fprintf(stderr, "Error no such file. [%s]\n", filename);
        return NULL;
    }
    int lineLen = MAX_CHAR_PER_LINE;
    line = new char[lineLen];
    assert(line != NULL);

    // get the row num of this file
    (*rowNum) = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        if (strtok(line, "\t\n") != 0)
        {
            ++ (*rowNum);
        }
    }

    rewind(infile);
    //get the col num in each line
    (*colObjNum) = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        if (strtok(line, " \t\n") != 0)      //escape the first char[id] in the file.
        {
            while (strtok(NULL, " ,\t\n") != NULL)
            {
                ++ (*colObjNum);
            }
            break;
        }
    }
    int i, j;
    double **objects = new double *[*rowNum];
    assert(objects != NULL);
    for (i = 0; i < (*rowNum); i++)
    {
        objects[i] = new double[*colObjNum];
        assert(objects[i] != NULL);
        memset(objects[i], 0, sizeof(double) * (*colObjNum));
    }
    rewind(infile);
    i = 0;
    while (fgets(line, lineLen, infile) != NULL)
    {
        if (strtok(line, " \t\n") == NULL) continue;
        for (j = 0; j < (*colObjNum); j ++)
        {
            objects[i][j] = atof(strtok(NULL, " ,\t\n"));
        }
        ++ i;
    }
    fclose(infile);
    delete line;
    line = NULL;
    return objects;
}

void KMeans::Init(double **objects, int rowNum, int colNum)
{
    m_rowNum = rowNum;
    m_colNum = colNum;
    double *tmpData = new double[m_colNum];
    assert(tmpData != NULL);

    for (int i = 0; i < m_clusterNum; i ++)
    {
        int indexAvg = m_rowNum/m_clusterNum;
        memset(tmpData, 0, sizeof(double) * m_colNum);
        int selectRowIndex = ((i == 0) ? (i*indexAvg) : (i*indexAvg-1));
        for (int j = 0; j < m_colNum; j ++)
        {
            tmpData[j] = objects[selectRowIndex][j];
        }
        m_Means[i] = new double[m_colNum];
        assert(m_Means[i] != NULL);
        memset(m_Means[i], 0, sizeof(double) * m_colNum);
        memcpy(m_Means[i], tmpData, sizeof(double) * m_colNum);
    }
    delete[] tmpData;
    tmpData = NULL;
}

// 随机选出m_clusterNum个点作为开始中心，赋予m_Means
void KMeans::Cluster(double **objects, int rowNum, int colObjNum, int *Label)
{
    Init(objects, rowNum, colObjNum);
    // 定义下一个聚类的中心
    double **next_Means = new double *[m_clusterNum];
    assert(next_Means != NULL);
    for (int i = 0; i < m_clusterNum; i ++)
    {
        next_Means[i] = new double[m_colNum];
        assert(next_Means[i] != NULL);
    }
    int *counts = new int[m_clusterNum];
    assert(counts != NULL);
    bool loop = true;
    double preCost = 0.0, curCost = 0.0;
    int curLabel = -1;
    int iterNum = 0, unChanged = 0;
    while (loop)
    {
        //初始化
        for(int i = 0; i < m_clusterNum; i ++)
        {
            memset(next_Means[i], 0, sizeof(double)*m_colNum);
        }
        memset(counts, 0, sizeof(int)*m_clusterNum);
        preCost = curCost;
        curCost = 0;
        //给所有数据分类
        for(int i = 0; i < m_rowNum; i ++)
        {
            curCost += GetLabel(objects[i], &curLabel);
            counts[curLabel] ++;
            for (int j = 0; j < m_colNum; j ++)
            {
                next_Means[curLabel][j] += objects[i][j];
            }
        }
        curCost /= m_rowNum;
        //重新计算m_Means[i]的值

        for (int i = 0; i < m_clusterNum; i ++)
        {
            if (counts[i] > 0)
            {
                for (int j = 0; j < m_colNum; j ++)
                {
                    next_Means[i][j] /= counts[i];
                }

            }
            memcpy(m_Means[i], next_Means[i], sizeof(int)*m_colNum);
        }
        //判断退出条件
        iterNum ++;
        if (fabs(curCost - preCost) < 1)
        {
            unChanged ++;
        }
        if (unChanged > 3 || iterNum > m_maxIterNum)
        {
            loop = false;
        }
    }
    // Save the labels
    for (int i = 0; i < m_rowNum; i ++)
    {
        GetLabel(objects[i], &curLabel);
        Label[i] = curLabel;
    }

    // delete the
    delete[] counts;
    counts = NULL;

    for (int i = 0; i < m_clusterNum; i ++)
    {
        delete[] next_Means[i];
        next_Means[i] = NULL;
    }
    delete[] next_Means;
    next_Means = NULL;
}

// Get the label of current array src and return the minDist
double KMeans::GetLabel(const double *src, int *label)
{
    double minDist = -1;
    for (int i = 0; i < m_clusterNum; i ++)
    {
        double tmp = CalcDistance(src, m_Means[i]);
        if (minDist == -1 || tmp < minDist)
        {
            minDist = tmp;
            *label = i;
        }
    }
    return minDist;
}

// Calculate the distance of two array[src[] and std_mean[]]
double KMeans::CalcDistance(const double *src, const double *std_mean)
{
    double sum = 0.0;
    for (int i = 0; i < m_colNum; i ++)
    {
        sum += (src[i] - std_mean[i]) * (src[i] - std_mean[i]);
    }
    return sqrt(sum);
}

void KMeans::file_write(const char *filename, const int *Labels)
{
    FILE *fileConn;
    char outFileName[1024];
    sprintf(outFileName, "%s.cluster_centers", filename);
    if (_DEBUG == true)
            cout << "Writing K = " << m_clusterNum <<" cluster centers to file \""<< outFileName << "\"." << endl;
    fileConn = fopen(outFileName, "w");
    int i, j;
    for (i = 0; i < m_clusterNum; i ++)
    {
        fprintf(fileConn, "%d\t", i+1);
        for (j = 0; j < m_colNum; j ++)
        {
            fprintf(fileConn, "%f ", m_Means[i][j]);
        }
        fprintf(fileConn, "\n");
    }
    fclose(fileConn);

    sprintf(outFileName, "%s.memberships", filename);
    if (_DEBUG == true)
            cout << "Begining writing N = " << m_rowNum <<" data Objects to file \""<< outFileName << "\"." << endl;
    fileConn = fopen(outFileName, "w");
    for (i = 0; i < m_rowNum; i ++)
    {
        fprintf(fileConn, "%d\t%d\n", i+1, Labels[i]);
    }
    fclose(fileConn);
}
void KMeans::OutCLusterMeans(int *Labels)
{
    cout << "\nThe cluster Number is: " << m_clusterNum  << ". rowNum: " << m_rowNum << ", colNum: " << m_colNum << endl;
    for (int i = 0; i < m_clusterNum; i ++)
    {
        for (int j = 0; j < m_colNum; j ++)
        {
            cout << m_Means[i][j] << "  ";
        }
        cout << endl;
    }
    cout << "\n\nAll cluster belong to as follow:\n";
    for (int i = 0; i < m_rowNum; i ++)
    {
        cout << "Data in line: " << i+1 << " belong to " << Labels[i] << " cluster." << endl;
    }
    cout << endl;
}
