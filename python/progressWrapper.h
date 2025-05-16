//
// Created by andrew on 11/28/2023.
//

#ifndef PLANC_PROGRESSWRAPPER_H
#define PLANC_PROGRESSWRAPPER_H

#include <nanobind/nanobind.h>


namespace nb = nanobind;

class Progress {
    nanobind::object wrappedbar;
    nanobind::callable wrappedbartick;
    int i_;
public:
    explicit Progress(unsigned long max, bool display_progress = true) {
        if (display_progress) {
            nb::module_ progressbar = nb::module_::import_("progressbar");
            nb::callable wrappedbarGen = progressbar.attr("progressbar");
            wrappedbar = wrappedbarGen(max);
            wrappedbartick = wrappedbar.attr("update");
            i_ = 0;
        }
    };
    void increment() {
        wrappedbartick(i_);
        i_++;
    };
};


#endif //PLANC_PROGRESSWRAPPER_H
