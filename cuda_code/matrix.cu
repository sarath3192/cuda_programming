#include <iostream>
#include <vector>

using namespace std;

const int N = 3; // Change N to any size you want

void multiplyMatrices(const vector<vector<int>> &A,
                      const vector<vector<int>> &B,
                      vector<vector<int>> &C) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

void printMatrix(const vector<vector<int>> &matrix) {
    for (const auto &row : matrix) {
        for (int val : row)
            cout << val << " ";
        cout << endl;
    }
}

int main() {
    vector<vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    vector<vector<int>> B = {
        {9, 8, 7},
        {6, 5, 4},
        {3, 2, 1}
    };

    vector<vector<int>> C(N, vector<int>(N, 0));

    multiplyMatrices(A, B, C);

    cout << "Result of A x B:" << endl;
    printMatrix(C);

    return 0;
}
