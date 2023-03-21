/* Copyright 2016 Ramakrishnan Kannan */

#ifndef BENCH_PLANC_BENCH_HPP_
#define BENCH_PLANC_BENCH_HPP_
#ifndef ARMA_DONT_USE_WRAPPER
#define ARMA_DONT_USE_WRAPPER
#endif
#define ONE_THREAD_MATRIX_SIZE 2000
#include "parsecommandline.hpp"
#include <string>
#include <vector>
namespace planc{

template <typename VT>
struct array_matrix {
    UWORD nrows = 0, ncols = 0;
    std::vector<VT> vals;
};

class NnlsParseCommandLine: public planc::ParseCommandLine
{
    private:
    int m_argc;
    char **m_argv;
    UWORD m_k;
    int m_num_it;
    int m_num_nodes;
    std::string m_Afile_name;
    std::string m_Bfile_name;
    std::string m_outputfile_name;
    bool nm_compute_error;

protected:
    NnlsParseCommandLine() : ParseCommandLine(m_argc, m_argv) {}

public:
    virtual std::string input_file_name() { return m_Afile_name; };
    virtual std::string output_file_name() { return m_outputfile_name; };
    std::string input_file_name_2() { return m_Bfile_name; };
    NnlsParseCommandLine(int argc, char **argv) : ParseCommandLine(m_argc, m_argv)
    {
        this->m_argc = argc;
        this->m_argv = argv;
        this->m_k = 20;
        this->m_num_it = 20;
        this->m_compute_error = 0;
    }
    void parseplancopts()
    {

        int opt, long_index;
        while ((opt = getopt_long(this->m_argc, this->m_argv,
                                  "e:i:j:k:o:p:s:t:h:n", plancopts,
                                  &long_index)) != -1)
        {
            switch (opt)
            {
            case 'e':
                this->m_compute_error = atoi(optarg);
                break;
            case 'i':
            {
                std::string temp = std::string(optarg);
                this->m_Afile_name = temp;
                break;
            }
            case 'j':
            {
                std::string temp = std::string(optarg);
                this->m_Bfile_name = temp;
                break;
            }
            case 'k':
                this->m_k = atoi(optarg);
                break;
            case 'o':
            {
                std::string temp = std::string(optarg);
                this->m_outputfile_name = temp;
                break;
            }
            case 't':
                this->m_num_it = atoi(optarg);
                break;
            case 'n':
                this->m_num_nodes = atoi(optarg);
                break;
            case 'h': // fall through intentionally
                print_usage();
                exit(0);
            case '?':
                INFO << "failed while processing argument:" << std::endl;
                print_usage();
                exit(EXIT_FAILURE);
            default:
                INFO << "failed while processing argument:" << std::endl;
                print_usage();
                exit(EXIT_FAILURE);
            }
        }

        // Input file must be given
        if (this->m_Afile_name.empty())
        {
            INFO << "Input not given." << std::endl;
            print_usage();
            exit(EXIT_FAILURE);
        }
        // Input file must be given
        if (this->m_Bfile_name.empty())
        {
            INFO << "Input not given." << std::endl;
            print_usage();
            exit(EXIT_FAILURE);
        }
    }
};
}
#endif // BENCH_PLANC_BENCH_HPP_
