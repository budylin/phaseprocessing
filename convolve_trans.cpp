#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations
#define EIGEN_DONT_PARALLELIZE
#include <iostream>
#include <Eigen/Dense>
#include <sys/time.h>
#include <functional>
#include <cmath>
#include <omp.h>
#include <functional>

using namespace Eigen;
using namespace std;

constexpr int around(int x)
{ return x / 120 * 120; }

const int data_in_packet = 120;
const int packet_size = data_in_packet + 8;
const int max_len = 32768 * data_in_packet ;

const int number_of_channel = around(2000);
const int packet_per_rgramm = number_of_channel / data_in_packet;
const int number_of_rgramms = max_len / number_of_channel;
const int number_of_packets = number_of_rgramms * packet_per_rgramm;
const int window_width = 64;

struct Packet
{
    unsigned char prefix[4];
    unsigned char data[data_in_packet];
    unsigned char suffix[4];
};

typedef Packet *Buffer;
typedef Matrix<float, Dynamic, 1> Phase;
typedef Matrix<float, window_width, 1> Window;


#define VL_EPSILON_F 1.19209290E-07F
#define VL_PI (float)M_PI
static inline float
vl_fast_atan2f (float y, float x) // err = 0.006 rad
{
  float angle, r ;
  float const c3 = 0.1821F ;
  float const c1 = 0.9675F ;
  float abs_y    = fabs (y) + VL_EPSILON_F ;

  if (x >= 0) {
    r = (x - abs_y) / (x + abs_y) ;
    angle = (float) (VL_PI / 4) ;
  } else {
    r = (x + abs_y) / (abs_y - x) ;
    angle = (float) (3 * VL_PI / 4) ;
  }
  angle += (c3*r*r - c1) * r ;
  return (y < 0) ? - angle : angle ;
}


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

typedef Matrix<float, Dynamic, Dynamic, ColMajor> Input;
typedef Matrix<complex<float>, Dynamic, Dynamic, ColMajor> Inputc;
typedef Matrix<float, number_of_channel, Dynamic> Output;

void process(Buffer x)
{

}

int main()
{
    Phase ph(number_of_rgramms);
    fillPhase(ph);
    Phase test(number_of_rgramms);
    fillPhase(test);

    Buffer x = new Packet[number_of_packets];
    fillBuffer(x, ph);

    float cos_val[] = {cos(0.), cos(2 * M_PI / 3), cos(4 * M_PI / 3)};
    float sin_val[] = {sin(0.), sin(2 * M_PI / 3), sin(4 * M_PI / 3)};

    Input data = Input::Zero(data_in_packet, number_of_packets);
    Input datacos = Input::Zero(number_of_channel, number_of_rgramms);
    Input datasin = Input::Zero(number_of_channel, number_of_rgramms);
    Output convcos = Output::Zero(number_of_channel, number_of_rgramms / 3);
    Output convsin = Output::Zero(number_of_channel, number_of_rgramms / 3);
    Output phase = Output::Zero(number_of_channel, number_of_rgramms / 3);
    Output amp = Output::Zero(number_of_channel, number_of_rgramms / 3);
    int N = 1;
    Window filter = Window(window_width);
    hemming(&filter);

    double t = omp_get_wtime();
    unsigned char *raw = (unsigned char*)x[0].data;
    internal::set_is_malloc_allowed(false);
    BufferMap y = BufferMap(raw, number_of_packets, data_in_packet);
    double t11 = omp_get_wtime();
    data.noalias() = y.transpose().cast<float>();
    data.resize(number_of_channel, number_of_rgramms);

    double t1 = omp_get_wtime();
    double t2, t3;
    #pragma omp parallel 
    {
        #pragma omp for
        for (int i = 0; i < data.cols(); i++)
        {
            datacos.col(i) = data.col(i) * cos_val[i % 3];
            data.col(i) *= sin_val[i % 3];
        }

        t2 = omp_get_wtime();
        #pragma omp for
        for(int i = 0; i < number_of_rgramms - window_width; i += 3)
        {
            convcos.col(i/3).noalias() =
                datacos.block<number_of_channel, window_width>(0, i)
                * filter;
            convsin.col(i/3).noalias() =
                data.block<number_of_channel, window_width>(0, i)
                * filter;
        }

        t3 = omp_get_wtime();
        #pragma omp for
        for(int i = 0; i < convsin.cols(); i++)
        {
            phase.col(i) = -convsin.col(i).binaryExpr(convcos.col(i),
                           ptr_fun(vl_fast_atan2f));
            amp.col(i) = convsin.col(i).binaryExpr(convcos.col(i),
                        [](float y, float x){return x * x + y * y ;});
        }
    }
    double t4 = omp_get_wtime();
    double t5 = omp_get_wtime();
    double dt = (t5 - t) / N;
    for (int i = 0; i < phase.cols() / 2 ; i+= phase.cols() / 30)
    {
        cout << i << " " << phase.row(1)[i] << " " << test[3 * i] << endl;
    }

    delete[]  x;
    cout << endl;
    cout << "x2:\t"     << t11 - t << endl;
    cout << "transp:\t" << t1 - t11 << endl;
    cout << "trig:\t"   << t2 - t1 << endl;
    cout << "conv:\t"   << t3 - t2 << endl;
    cout << "atan2:\t"  << t4 - t3 << endl;
    cout << "hypot:\t"  << t5 - t4 << endl;
    cout << "omp_get_wtime:\t" << dt << " s" << endl;
    cout << "freq:\t" << 1 / dt << " BPS" << endl;
}

