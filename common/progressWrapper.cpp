 //
// Created by andrew on 11/28/2023.
//

#include "progressWrapper.h"

#include <memory>


Progress::Progress(unsigned long max, bool display_progress) {
#ifndef NO_INDICATORS
    using namespace indicators;
    wrappedBar = std::make_unique<BlockProgressBar>(
            option::MaxProgress{max},
            option::ForegroundColor{Color::white},
            option::ShowElapsedTime{display_progress},
            option::ShowRemainingTime{display_progress},
            option::ShowPercentage{display_progress});
#endif
}
    void Progress::increment() const {
#ifndef NO_INDICATORS
        wrappedBar->tick();
#endif
        }
