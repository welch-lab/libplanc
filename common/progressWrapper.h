//
// Created by andrew on 11/28/2023.
//

#ifndef PLANC_PROGRESSWRAPPER_H
#define PLANC_PROGRESSWRAPPER_H

#include <cstdint>
#ifndef NO_INDICATORS
#include <indicators/block_progress_bar.hpp>
#endif

class Progress {
private:
#ifndef NO_INDICATORS
    std::unique_ptr<indicators::BlockProgressBar> wrappedBar;
#endif
public:
    explicit Progress(unsigned long max, bool display_progress = true);
    void increment() const;
};


#endif //PLANC_PROGRESSWRAPPER_H
