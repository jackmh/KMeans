#ifndef KMEANS_H_INCLUDED
#define KMEANS_H_INCLUDED

#define MAX_CHAR_PER_LINE 128
#define MAX_BUFFERSIZE 1024
#define _DEBUG true

class KMeans
{
public:
    KMeans(int clusterNum = 4, int rowNum = 2, int colNum = 2);
    ~KMeans();

    double** file_read(const char *filename, int  *rowNum, int  *colObjNum);
    void file_write(const char *filename, const int *Labels);

    // 随机选出m_clusterNum个点作为开始中心，赋予m_Means
    void Init(double **objects, int rowNum, int colNum);
    void Cluster(double **objects, int rowNum, int colObjNum, int *Label);
    void OutCLusterMeans(int *Labels);

private:
    int m_rowNum;       // the row line number
    int m_colNum;       // the column object number
    int m_clusterNum;   // how many cluster
    double **m_Means;   //The center of KMeans with m_clusterNum*m_rowNum matrix

    int m_maxIterNum;   //The max iterative Number
    double m_endError;

    double GetLabel(const double *src, int *label);
    double CalcDistance(const double *src, const double *std_mean);
};

#endif // KMEANS_H_INCLUDED
