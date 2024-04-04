#include <QuEST.h>

#include <assert.h>
#include <math.h>
#include <stdlib.h>

qreal H;

void initH(Qureg qureg)
{
    int numQubits = qureg.numQubitsInStateVec;
    H = powf(1.f / sqrt(2.f), (qreal)numQubits);
}

ull *init1DGraph(Qureg qureg)
{
    return (ull *)calloc(qureg.numQubitsInStateVec, sizeof(ull));
}

void addEdgeTo1DGraph(Qureg qureg, ull *graph, int qubit1, int qubit2)
{
    int numQubits = qureg.numQubitsInStateVec;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2)
    {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1] |= 1ull << qubit2;
}

void free1DGraph(ull *graph)
{
    free(graph);
}

bool *init2DGraph(Qureg qureg)
{
    int numQubits = qureg.numQubitsInStateVec;
    return (bool *)calloc(numQubits * numQubits, sizeof(bool));
}

qreal *initWeights(Qureg qureg)
{
    int numQubits = qureg.numQubitsInStateVec;
    return (qreal *)malloc(numQubits * numQubits * sizeof(qreal));
}

void freeWeights(qreal *weights)
{
    free(weights);
}

void addEdgeTo2DGraph(Qureg qureg, bool *graph, qreal *weights, int qubit1, int qubit2, qreal weight)
{
    int numQubits = qureg.numQubitsInStateVec;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2)
    {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1 * numQubits + qubit2] = true;
    weights[qubit1 * numQubits + qubit2] = weight;
}

void free2DGraph(bool *graph)
{
    free(graph);
}