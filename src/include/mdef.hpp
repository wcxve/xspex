#ifndef XSPEX_MDEF_HPP_
#define XSPEX_MDEF_HPP_

// Model definition macro that takes model name and parameter count
// This file contains the central definition of all XSPEC models
// Format: ENTRY(model_name, n_params)
//
// To add a new model:
// 1. Add ENTRY(your_model_name, n_params) below
// 2. Ensure the corresponding C_your_model_name function exists in XSPEC
// 3. Rebuild the project

// Additive and multiplicative models
#define XSPEC_USER_MODELS                                      \
    /* Add your custom additive/multiplicative models here: */ \
    /* Example: ENTRY(mymodel, 3)                           */ \
    /* Leave empty if no custom models                      */

// Convolution model definitions
#define XSPEC_USER_CON_MODELS                                  \
    /* Add your custom convolution models here:             */ \
    /* Example: ENTRY(myconvmodel, 2)                       */ \
    /* Leave empty if no custom models                      */

#ifndef XSPEC_MODEL_LIST
#define XSPEC_MODEL_LIST
#endif
#ifndef XSPEC_CON_MODEL_LIST
#define XSPEC_CON_MODEL_LIST
#endif

#define XSPEC_MODELS XSPEC_MODEL_LIST XSPEC_USER_MODELS
#define XSPEC_CON_MODELS XSPEC_CON_MODEL_LIST XSPEC_USER_CON_MODELS

// Auto-calculate model count
static constexpr int count_xspec_models()
{
    int count = 0;
#define ENTRY(name, n_params) ++count;
    XSPEC_MODELS
    XSPEC_CON_MODELS
#undef ENTRY
    return count;
}

#define XSPEC_MODEL_COUNT count_xspec_models()

#endif  // XSPEX_MDEF_HPP_
