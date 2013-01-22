#include <iostream>
#include <Eigen/Dense>
#include <sys/time.h>
#include <functional>
#include <cmath>
#include <omp.h>

using namespace Eigen;
using namespace std;

constexpr int around(int x)
{ return x / 120 * 120; }

const int data_in_packet = 128;
const int packet_size = data_in_packet + 8;
const int max_len = 32000 * data_in_packet ;

const int number_of_channel = around(2000);
const int packet_per_rgramm = number_of_channel / data_in_packet;
const int number_of_rgramms = max_len / number_of_channel;
const int number_of_packets = number_of_rgramms * packet_per_rgramm;
const int window_width = 48;

typedef Matrix<float, Dynamic, number_of_channel, ColMajor> Input;
typedef Matrix<float, Dynamic, number_of_channel> Output;
typedef Matrix<float, window_width, 1> Window;

struct Packet
{
    unsigned char prefix[4];
    unsigned char data[data_in_packet];
    unsigned char suffix[4];
};

typedef Packet Buffer[number_of_packets];
typedef Matrix<float, number_of_rgramms, 1> Phase;

void hemming(Window *vec)
{
    Window &ref = *vec;
    int M = ref.rows();
    for (int i = 0; i < M; i++)
        ref(i) = 0.54 - 0.46 * cos( 2 * M_PI * i / (M - 1));
}

void fillBuffer(Buffer x, const Phase ph)
{
    const float A = 7.5;
    const float B = 7.5;
    /* I = A^2 + B^2 + 2ABcos(2pik/3 + phi) */
    int i;
    for (i = 0; i < number_of_packets; i++)
    {
        int rgramm = i / packet_per_rgramm;
        float phi_k = ph[rgramm];
        float Ik = A * A + B * B + 2 * A * B * cos(2 * M_PI / 3 * rgramm + phi_k);
        for (int j = 0; j < 4; j++)
        {
            x[i].prefix[j] = 113;
            x[i].suffix[j] = 131;
        }
        for(int j = 0; j < data_in_packet; j++)
        {
            x[i].data[j] = static_cast<unsigned char>(Ik);
        }
    }
}

void fillPhase(Phase &ph)
{
    const int phi_period = number_of_rgramms / 2;
    for (int i = 0; i < number_of_rgramms; i++)
        ph[i] = 1 + 1 * cos(2 * M_PI * i / phi_period);

}
//The default in Eigen is column-major
typedef Matrix<unsigned char, Dynamic, Dynamic, RowMajor> Data;
typedef Map<Data, Unaligned, Stride<packet_size, 0>> BufferMap;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> Refs;

int main()
{
    Phase ph(number_of_rgramms);
    fillPhase(ph);
    Phase test(number_of_rgramms);
    fillPhase(test);

    Buffer x;
    fillBuffer(x, ph);

    float cos_val[] = {cos(0.), cos(2 * M_PI / 3), cos(4 * M_PI / 3)};
    float sin_val[] = {sin(0.), sin(2 * M_PI / 3), sin(4 * M_PI / 3)};
    ArrayXf coss(number_of_rgramms);
    ArrayXf sins(number_of_rgramms);
    for (int i = 0; i < number_of_rgramms; i++)
    {
        coss[i] = cos_val[i % 3];
        sins[i] = sin_val[i % 3];
    }

    Input datacos = Input::Zero(number_of_rgramms, number_of_channel);
    Input datasin = Input::Zero(number_of_rgramms, number_of_channel);
    Output convcos = Output::Zero((number_of_rgramms + 2) / 3, number_of_channel);
    Output convsin = Output::Zero((number_of_rgramms + 2) / 3, number_of_channel);
    Output phase =   Output::Zero((number_of_rgramms + 2) / 3, number_of_channel);
    int N = 1;

    unsigned char *raw = (unsigned char*)x[0].data;
    BufferMap y = BufferMap(raw, number_of_packets, data_in_packet);
    Refs yy = y.cast<float>();
    yy.resize(number_of_rgramms, number_of_channel);

    Input data = yy;

    data.resize(number_of_rgramms, number_of_channel);

    Window filter = Window(window_width);
    hemming(&filter);


    double t = omp_get_wtime();
    datacos = data.array().colwise() * coss;
    datasin = data.array().colwise() * sins;

    double t2 = omp_get_wtime();
    #pragma omp parallel for
    for(int i = 0; i < number_of_rgramms - window_width - 1; i += 3)
    {
        convcos.row(i/3) = filter.transpose() *
                           datacos.block<window_width, number_of_channel>(i, 0);
        convsin.row(i/3) = filter.transpose() *
                           datasin.block<window_width, number_of_channel>(i, 0);
    }

    double t3 = omp_get_wtime();
/*
    for (int i = 0; i < number_of_channel; i++)
        for (int j = 0; j < (number_of_rgramms + 2) / 3; j++)
            phase(i, j) = atan2f(convsin(i, j), convcos(i, j));
slower by factor of 3.3 */

    double t4 = omp_get_wtime();
    phase = convsin.binaryExpr(convcos, ptr_fun(atan2f));
    double t5 = omp_get_wtime();
    double dt = (omp_get_wtime() - t) / N;

    for (int i = 0; i < phase.rows() / 2 ; i++)
    {
    cout << i << " " << phase.col(1)[i] << " " << test[3 * i] << endl;
    }

    cout << "cos, sin: " << t2 - t << endl;
    cout << "conv: " << t3 - t2 << endl;
    cout << "atan2: " << t4 - t3 << endl;
    cout << "atan2: " << t5 - t4 << endl;
    cout << "omp_get_wtime : " << dt << " s" << endl;
    cout << "freq : " << 1 / dt << " BPS" << endl;
}

