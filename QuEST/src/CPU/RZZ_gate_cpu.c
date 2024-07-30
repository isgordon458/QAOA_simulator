#include <QuEST.h>

extern qreal H;

void rotationCompressionWeighted(Qureg qureg, qreal angle, bool *graph, qreal *weights, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

#pragma omp parallel for
    for (ull q = 0; q < numAmps; q++)
    {
        qreal sum = 0;
        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            for (int j = i + 1; j < numQubits; j++)
            {
                int idx = i * numQubits + j;
                if (graph[idx])
                {
                    if (((q >> i) & 0b1) ^ ((q >> j) & 0b1))
                        sum += weights[idx];
                    else
                        sum -= weights[idx];
                }
            }
        }

        qreal contracted_rzz[2];

        sincos(sum * angle / 2.0, &contracted_rzz[1], &contracted_rzz[0]);

        if (isFirstLayer)
        {
            qureg.stateVec.real[q] = contracted_rzz[0] * H;
            qureg.stateVec.imag[q] = contracted_rzz[1] * H;
        }
        else
        {
            qreal in[2];
            in[0] = qureg.stateVec.real[q];
            in[1] = qureg.stateVec.imag[q];
            qureg.stateVec.real[q] = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
            qureg.stateVec.imag[q] = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
        }
    }
}

void rotationCompressionUnweighted(Qureg qureg, qreal angle, ull *graph, bool isFirstLayer)
{
    int numQubits = qureg.numQubitsInStateVec;
    ull numAmps = 1ull << qureg.numQubitsInStateVec;

    int numGates = 0;
    for (int i = 0; i < numQubits; i++)
        numGates += __builtin_popcountll(graph[i]);

#pragma omp parallel for
    for (ull q = 0; q < numAmps; q += 1)
    {
        int Cp = 0;
        // count number of 1
        for (int i = 0; i < numQubits; i++)
        {
            ull strxor;
            ull msk = -((q >> i) & 1LL);
            strxor = graph[i] & (q ^ msk);
            Cp += __builtin_popcountll(strxor);
        }

        qreal contracted_rzz[2];
        sincos((2 * Cp - numGates) * angle / 2, &contracted_rzz[1], &contracted_rzz[0]);

        if (isFirstLayer)
        {
            qureg.stateVec.real[q] = contracted_rzz[0] * H;
            qureg.stateVec.imag[q] = contracted_rzz[1] * H;
        }
        else
        {
            qreal in[2];
            in[0] = qureg.stateVec.real[q];
            in[1] = qureg.stateVec.imag[q];
            qureg.stateVec.real[q] = in[0] * contracted_rzz[0] - in[1] * contracted_rzz[1];
            qureg.stateVec.imag[q] = in[0] * contracted_rzz[1] + in[1] * contracted_rzz[0];
        }
    }
}