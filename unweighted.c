#include <QuEST.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils.h"

int main(int argc, char *argv[]) {

    const qreal gamma = 0.5 * M_PI;
    const qreal beta = 0.6 * M_PI;

    int P;  // number of rounds

    if (argc >= 2) {
        P = atoi(argv[1]);
    } else {
        P = 1;
    }

    printf("%6s, %15s, %22s\n", "qubits", "time (us)", "stateVector[0]");

    for (int num_qubits = 2; num_qubits <= 30; num_qubits++) {

        QuESTEnv env = createQuESTEnv();
        Qureg qubits = createQureg(num_qubits, env);

        ull *graph = init1DGraph(qubits);

        // Fully connected graph
        for (int i=0; i<num_qubits; i++)
            for (int j=i+1; j<num_qubits; j++)
                addEdgeTo1DGraph(qubits, graph, i, j);  // add edge that connects node i and node j

        // Initialize for Launch control
        initH(qubits);

        MEASURET_START;

        for (int p=0; p<P; p++) {
            
            rotationCompressionUnweighted(qubits, gamma, graph, p==0);

            for (int i=0; i<num_qubits; i++) {
                rotateX(qubits, i, beta);
            }
        }

        syncQuESTEnv(env);
        copyStateFromGPU(qubits);

        MEASURET_END;

        printf("%6d, %15lu, % .6f + % .6fi\n", num_qubits, diff, qubits.stateVec.real[0], qubits.stateVec.imag[0]);
        fflush(stdout);
        free1DGraph(graph);

        // unload QuEST
        destroyQureg(qubits, env); 
        destroyQuESTEnv(env);
    }

    return 0;
}