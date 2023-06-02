#include "../common/inmf.hpp"
#include "../common/utils.hpp"
#include <memory>
#include <vector>

int main(int argc, char* argv[]) {
	arma::mat tempMat = arma::zeros(10, 10);
	arma::mat* tempMatptr = &tempMat;
	std::unique_ptr<arma::mat> testPtr = std::unique_ptr<arma::mat>(tempMatptr);
	std::vector<std::unique_ptr<arma::mat>> ptrvec;
	ptrvec.push_back(std::move(testPtr));
	planc::INMF<arma::mat> testINMF = planc::INMF<arma::mat>(ptrvec, 10u, 4.);
}