#pragma once
#include <cstdint>

namespace tl {

    // THE UNIT SYSTEM
    // Length: 1 unit = 1 femtometer (1e-15 m)
    // Time:   1 unit = 1 femtosecond (1e-15 s)
    // Mass:   1 unit = 1 dalton (scaled)
    // Energy: 1 unit = 1 kilojoule/mole (scaled)

    // The Fixed Point Type
    // Range: +/- 9.22e18 units.
    typedef int64_t fixed_t;

    // Scale Factor (Implicit in the type, but noted for conversion)
    constexpr double SCALE_FACTOR = 1e9; 

    // Helper struct for 3D coordinates
    struct int3_vec {
        fixed_t x;
        fixed_t y;
        fixed_t z;
    };
}
