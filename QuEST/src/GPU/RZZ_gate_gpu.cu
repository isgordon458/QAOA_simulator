#include <QuEST.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

__managed__ qreal H;

void initH(Qureg qureg) {
    int numQubits = qureg.numQubitsInStateVec;
    H = pow(1./sqrt(2.), numQubits);
}

ull* init1DGraph(Qureg qureg) {
   ull *graph;
   cudaMallocManaged(&graph, qureg.numQubitsInStateVec * sizeof(ull));
   memset(graph, 0, qureg.numQubitsInStateVec * sizeof(ull));
   return graph;
}

void free1DGraph(ull* graph) {
    assert(cudaFree(graph) == cudaSuccess);
}

void addEdgeTo1DGraph(Qureg qureg, ull* graph, int qubit1, int qubit2) {
    int numQubits = qureg.numQubitsInStateVec;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1] |= 1ull << qubit2;
}

bool* init2DGraph(Qureg qureg) {
    int numQubits = qureg.numQubitsInStateVec;
    bool *graph;
    assert(cudaMallocManaged(&graph, numQubits*numQubits * sizeof(bool)) == cudaSuccess);
    return graph;
}

qreal* initWeights(Qureg qureg) {
    int numQubits = qureg.numQubitsInStateVec;
    qreal *weights;
    assert(cudaMallocManaged(&weights, numQubits*numQubits * sizeof(qreal)) == cudaSuccess);
    return weights;
}

void addEdgeTo2DGraph(Qureg qureg, bool* graph, qreal *weights, int qubit1, int qubit2, qreal weight) {
    int numQubits = qureg.numQubitsInStateVec;
    assert(qubit1 < numQubits);
    assert(qubit2 < numQubits);
    if (qubit1 > qubit2) {
        int tmp = qubit1;
        qubit1 = qubit2;
        qubit2 = tmp;
    }
    graph[qubit1*numQubits+qubit2] = 1;
    weights[qubit1*numQubits+qubit2] = weight;
}

void free2DGraph(bool* graph) {
    assert(cudaFree(graph) == cudaSuccess);
}

void freeWeights(qreal *weights)
{
    assert(cudaFree(weights) == cudaSuccess);
}

__global__
void _rotationCompressionWeighted(qreal *real, qreal *imag, int numQubits, qreal angle, bool *graph, qreal *weights, int numGates, bool isFirstLayer){

    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull b = tidx;

    qreal sum = 0;
    // count number of 1
    for (int i = 0; i < numQubits; i++) {
        for (int j = i + 1; j < numQubits; j++) {
            int idx = i*numQubits+j;
            if (graph[idx]) {
                if (((b>>i)&0b1) ^ ((b>>j)&0b1))
                    sum += weights[idx];
                else
                    sum -= weights[idx];
            }
        }
    }

    qreal contracted_rzz[2];
    sincos(sum*angle/2, &contracted_rzz[1], &contracted_rzz[0]);

    if (isFirstLayer) {
        real[tidx] = contracted_rzz[0] * H;
        imag[tidx] = contracted_rzz[1] * H;
    } else {
        qreal in[2];
        in[0] = real[tidx];
        in[1] = imag[tidx];
        real[tidx] = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
        imag[tidx] = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
    }
}

void rotationCompressionWeighted(Qureg qureg, qreal angle, bool *graph, qreal *weights, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << numQubits;

    ull grid = 1;
    ull block = 128;

    if (numAmps > block)
        grid = numAmps / 128;
    else
        block = numAmps;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++) {
        for (int j = i + 1; j < numQubits; j++) {
            numGates++;
        }
    }

    _rotationCompressionWeighted<<<grid, block>>>(qureg.deviceStateVec.real, qureg.deviceStateVec.imag, qureg.numQubitsInStateVec, angle, graph, weights, numGates, isFirstLayer);
}

__global__
void _rotationCompressionUnweighted(qreal *real, qreal *imag, int numQubits, qreal angle, ull *graph, int numGates, bool isFirstLayer) {

    ull tidx = blockIdx.x * blockDim.x + threadIdx.x;
    ull b = tidx;
    
    int Cp = 0;
    // count number of 1
    for (int i = 0; i < numQubits; i++) {
        ull strxor;
        ull msk = -((b >> i) & 1LL);
        strxor = graph[i] & (b ^ msk);
        Cp += __popcll(strxor);
    }

    qreal contracted_rzz[2];
    sincos((2*Cp-numGates)*angle/2, &contracted_rzz[1], &contracted_rzz[0]);

    if (isFirstLayer) {
        real[tidx] = contracted_rzz[0] * H;
	    imag[tidx] = contracted_rzz[1] * H;
    } else {
        qreal in[2];
        in[0] = real[tidx];
        in[1] = imag[tidx];
        real[tidx] = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
        imag[tidx] = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
    }
}

void rotationCompressionUnweighted(Qureg qureg, qreal angle, ull *graph, bool isFirstLayer) {

    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << numQubits;

    ull grid = 1;
    ull block = 128;

    if (numAmps > block)
        grid = numAmps / 128;
    else
        block = numAmps;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++)
        numGates += __builtin_popcountll(graph[i]);

    _rotationCompressionUnweighted<<<grid, block>>>(qureg.deviceStateVec.real, qureg.deviceStateVec.imag, qureg.numQubitsInStateVec, angle, graph, numGates, isFirstLayer);
}