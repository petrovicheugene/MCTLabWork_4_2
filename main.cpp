//=====================================================
#include <iostream>
#include <omp.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <regex>
#include <iomanip>
//=====================================================
using namespace std;

typedef vector<vector<double>> Matrix;

bool loadLESDataFromFile();
void trimLine(string& line);
vector<double> parseLine(const string& line);
void solveLES();
unsigned int findMaxElementRow(Matrix matrix, unsigned int column);
Matrix triangulateMatrix(const Matrix& matrix, int* pSwapCount);
void outputSolutionToFile();
void outputSolutionToDisplay();
void outputMatrixToDisplay(const Matrix& matrix);

static Matrix LES;
static vector<double> X;
static string inFileName = "./LES_data.txt";
static string outFileName = "./LES_solution.txt";

//=====================================================
int main()
{
    cout << "Solving a system of equations." << endl;

    if(loadLESDataFromFile())
    {
        outputMatrixToDisplay(LES);
        solveLES();
        outputSolutionToDisplay();
        outputSolutionToFile();
    }

    getchar();
    return 0;
}
//=====================================================
bool loadLESDataFromFile()
{
    cout << "LES initialization from file \"" << inFileName << "\"" << endl;
    cout << endl;

    // open file
    ifstream file(inFileName);
    if (!file.is_open())
    {
        cout << "Unable to open data file!" << endl;
        return false;
    }

    string line;
    string matrixName;
    vector<double> values;
    while (getline(file, line))
    {
        trimLine(line);
        if (line.empty()) continue;
        // Matrix initialization
        values = parseLine(line);
        if (values.empty())
        {
            continue;
        }

        LES.push_back(values);
    }

    file.close();
    return true;
}
//=====================================================
void trimLine(string& line)
{
    line.erase(line.find_last_not_of(" \n\r\t") + 1);
    line.erase(0, line.find_first_not_of(" "));
}
//=====================================================
vector<double> parseLine(const string& line)
{
    std::regex regex{ R"([\s,]+)" }; // split by space and comma
    std::sregex_token_iterator it{ line.begin(), line.end(), regex, -1 };
    std::vector<std::string> words{ it, {} };
    vector<double> values;

    for (size_t i = 0; i < words.size(); i++)
    {
        values.push_back(stod(words.at(i)));
    }
    return values;
}
//=====================================================
void solveLES()
{
    Matrix tLES = triangulateMatrix(LES, nullptr);

    unsigned int row_count = tLES.size();
    unsigned int free_member_col = row_count;
    X.resize(row_count);

    for (int row = static_cast<int>(row_count) - 1; row >= 0; row--)
    {
        double right_part = tLES[static_cast<unsigned int>(row)][free_member_col];
        for (int col = static_cast<int>(free_member_col) - 1; col > row; col--)
        {
            right_part -= X[static_cast<unsigned int>(col)]
                    * tLES[static_cast<unsigned int>(row)][static_cast<unsigned int>(col)];
        }
        X[static_cast<unsigned int>(row)] = right_part
                / tLES[static_cast<unsigned int>(row)][static_cast<unsigned int>(row)];
    }
}
//=====================================================
unsigned int findMaxElementRow(Matrix matrix, unsigned int column)
{
    unsigned int max_row_index = column;
    double max = abs(matrix[column][column]);
    unsigned int matrix_size = matrix.size();
    bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
    {
        double local_max = max;
        unsigned int local_max_row_index = max_row_index;
#pragma omp for
        for (unsigned int i = column + 1; i < matrix_size; i++)
        {
            double value = abs(matrix[i][column]);
            if (value > max)
            {
                local_max = value;
                local_max_row_index = i;
            }
        }

#pragma omp critical
        {
            if (max < local_max)
            {
                max = local_max;
                max_row_index = local_max_row_index;
            }
        }
    }
    return max_row_index;
}
//=====================================================
Matrix triangulateMatrix(const Matrix& matrix, int* pSwapCount)
{
    // create and init new Matrix
    Matrix new_matrix = matrix;

    unsigned int matrix_size = new_matrix.size();
    unsigned int matrix_column_size = new_matrix[0].size();

    int swapCount = 0;
    for (unsigned int i = 0; i < matrix_size - 1; i++)
    {
        unsigned int maxElementRow = findMaxElementRow(new_matrix, i);

        if (i != maxElementRow)
        {
            swap(new_matrix[i], new_matrix[maxElementRow]);
            ++swapCount;
        }

        bool parallelAvailable = !static_cast<bool>(omp_in_parallel());
#pragma omp parallel if (parallelAvailable)
        {
#pragma omp for
            for (unsigned int row = i + 1; row < matrix_size; row++)
            {
                double f = -new_matrix[row][i] / new_matrix[i][i];
                for (unsigned int k = i; k < matrix_column_size; ++k)
                {
                    new_matrix[row][k] += new_matrix[i][k] * f;
                }
            }
        }
    }
    if (pSwapCount != nullptr)
    {
        *pSwapCount = swapCount;
    }
    return new_matrix;
}
//=====================================================
void outputSolutionToFile()
{
    if (X.empty())
    {
        return;
    }

    ofstream file(outFileName);
    if (!file.is_open())
    {
        cout << "Unable to open data file" << endl;
        return;
    }

    file << "LES Solution:" << endl;
    file << fixed << setprecision(2);
    for (unsigned int i = 0; i < X.size(); i++)
    {
        file << "X" << i + 1 << ": " << X[i] << endl;
    }

    file.close();
}
//=====================================================
void outputSolutionToDisplay()
{
    cout << "LES Solution:" << endl;

    for (unsigned int i = 0; i < X.size(); i++)
    {
        printf("X%d: %8.3f\n", i, X[i]);
    }
}
//=====================================================
void outputMatrixToDisplay(const Matrix& matrix)
{
    if (matrix.empty())
    {
        return;
    }

    unsigned int rows = matrix.size();
    unsigned int columns = matrix.at(0).size();

    for (unsigned int row = 0; row < rows; row++)
    {
        for (unsigned int col = 0; col < columns; col++)
        {
            printf("%9.2f", matrix[row][col]);
        }
        cout << endl;
    }
    cout << endl;
}
//=====================================================
