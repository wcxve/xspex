#ifndef __XSPEX_HPP__
#define __XSPEX_HPP__

#include <iostream>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <XSFunctions/functionMap.h>
#include <XSFunctions/funcWrappers.h>
#include <XSFunctions/Utilities/FunctionUtility.h>
#include <XSFunctions/Utilities/funcType.h>  // xsccCall and the like
#include <XSFunctions/Utilities/xsFortran.h>  // needed for FNINIT
#include <XSUtil/Utils/XSutility.h>

namespace py = pybind11;

typedef py::buffer_info BufferInfo;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> DoubleArray;
typedef py::array_t<float, py::array::c_style | py::array::forcecast> FloatArray;

namespace xspex {
    // Initialize the Xspec interface.
    void init() {
        if (!getenv("HEADAS")) {
            throw std::runtime_error("The HEADAS environment variable is not set!");
        }

        // FNINIT is a bit chatty, so hide the stdout buffer for this call.
        std::ostringstream local;
        auto cout_buff = std::cout.rdbuf();
        std::cout.rdbuf(local.rdbuf());
        try {
            FNINIT();
        } catch(...) {
            std::cout.rdbuf(cout_buff);
            throw std::runtime_error("Unable to initialize Xspec model library\n" + local.str());
        }

        // Get back original std::cout
        std::cout.rdbuf(cout_buff);
    }


    // Get the version of Xspec
	string get_version() { return XSutility::xs_version(); };


    // Get & set the chatter level
    int get_chatter() { return FunctionUtility::xwriteChatter(); }
    void set_chatter(int i) { FunctionUtility::xwriteChatter(i); }


    // Xspec abund
    string get_abundance() { return FunctionUtility::ABUND(); }
    void set_abundance(const string &value) { FunctionUtility::ABUND(value); }

    float abundance_by_name(const string &value) {
        // We check to see if an error was written to stderr to identify when the
        // input was invalid. This is not great!
        std::ostringstream local;
	    auto cerr_buff = std::cerr.rdbuf();
	    std::cerr.rdbuf(local.rdbuf());

	    // Assume this can not throw an error
	    auto answer = FunctionUtility::getAbundance(value);

	    std::cerr.rdbuf(cerr_buff);
	    if (local.str() != "") { throw py::key_error(value); }

	    return answer;
    }
    float abundance_by_z(const size_t Z) {
	    if (Z < 1 || Z > FunctionUtility::NELEMS()) {
            std::ostringstream emsg;
            emsg << Z;
            throw py::index_error(emsg.str());
	    }

	    return FunctionUtility::getAbundance(Z);
    }

    string element_name_by_z (const size_t Z) { return FunctionUtility::elements(Z - 1); }


    // Assume this is not going to change within a session!
    // Also we assume that this can be called without callnig FNINIT.
    size_t number_elements = FunctionUtility::NELEMS();


    // Xspec xsect
    string get_cross_section() { return FunctionUtility::XSECT(); }
    void set_cross_section(const string &value) { FunctionUtility::XSECT(value); }


    // Cosmology settings: I can not be bothered exposing the per-setting values.
    std::map<std::string, float> get_cosmology() {
	    std::map<std::string, float> answer;
	    answer["H0"] = FunctionUtility::getH0();
	    answer["q0"] = FunctionUtility::getq0();
	    answer["lambda0"] = FunctionUtility::getlambda0();
	    return answer;
	}

    void set_cosmology(float H0, float q0, float lambda0) {
	    FunctionUtility::setH0(H0);
	    FunctionUtility::setq0(q0);
	    FunctionUtility::setlambda0(lambda0);
	}




    // Check the number of parameters
    void validate_par_size(const int NumPars, const int got) {
        if (NumPars == got)
            return;

        std::ostringstream err;
        err << "Expected " << NumPars << " parameters but sent " << got;
        throw std::runtime_error(err.str());
    }


    // Provide a useful error message if the sizes don't match
    void validate_grid_size(const int energySize, const int modelSize) {
        if (energySize == modelSize + 1)
            return;

        std::ostringstream err;
        err << "Energy grid size must be 1 more than model: "
            << "energies has " << energySize << " elements and "
            << "model has " << modelSize << " elements";
        throw py::value_error(err.str());
    }


    void float_to_double(float *in, double *out, int size) {
        for (int i = 0; i < size; ++i) {
            out[i] = static_cast<double>(in[i]);
        }
    }


    void double_to_float(double *in, float *out, int size) {
        for (int i = 0; i < size; ++i) {
            out[i] = static_cast<float>(in[i]);
        }
    }


    template <typename T, xsccCall model, int NumPars>
    py::array_t<T> wrapper_C(
        const DoubleArray pars,
        const DoubleArray energyArray,
        const int spectrumNumber,
        const string initStr
    ) {
        BufferInfo pbuf = pars.request();
        BufferInfo ebuf = energyArray.request();
        if (pbuf.ndim != 1 || ebuf.ndim != 1) {
            throw py::value_error("pars and energyArray must be 1D");
        }

        validate_par_size(NumPars, pbuf.size);

        if (ebuf.size < 2) {
            throw py::value_error("Expected at least 2 bin edges");
        }

        // Should we force spectrumNumber >= 1?
        // We shouldn't be able to send in an invalid initStr so do not bother checking.

        const int nelem = ebuf.size - 1;

        // Can we easily zero out the arrays?
        auto result = py::array_t<double>(nelem);
        auto errors = std::vector<double>(nelem);

        BufferInfo obuf = result.request();

        double *pptr = static_cast<double *>(pbuf.ptr);
        double *eptr = static_cast<double *>(ebuf.ptr);
        double *optr = static_cast<double *>(obuf.ptr);

        model(eptr, nelem, pptr, spectrumNumber, optr, errors.data(), initStr.c_str());
        return result;
    }


    template <typename T, xsf77Call model, int NumPars>
    py::array_t<T> wrapper_f(
        const FloatArray pars,
        const FloatArray energyArray,
        const int spectrumNumber
    ) {
        BufferInfo pbuf = pars.request();
        BufferInfo ebuf = energyArray.request();

        if (pbuf.ndim != 1 || ebuf.ndim != 1){
            throw py::value_error("pars and energyArray must be 1D");
        }

        validate_par_size(NumPars, pbuf.size);

        if (ebuf.size < 2) {
            throw py::value_error("Expected at least 2 bin edges");
        }

        const int nelem = ebuf.size - 1;

        // Can we easily zero out the arrays?
        auto result = py::array_t<float>(nelem);
        auto errors = std::vector<float>(nelem);

        BufferInfo obuf = result.request();

        float *pptr = static_cast<float *>(pbuf.ptr);
        float *eptr = static_cast<float *>(ebuf.ptr);
        float *optr = static_cast<float *>(obuf.ptr);

        model(eptr, nelem, pptr, spectrumNumber, optr, errors.data());
        return result;
    }


    template <typename T, xsF77Call model, int NumPars>
    py::array_t<T> wrapper_F(
        const DoubleArray pars,
        const DoubleArray energyArray,
        const int spectrumNumber
    ) {
        BufferInfo pbuf = pars.request();
        BufferInfo ebuf = energyArray.request();

        if (pbuf.ndim != 1 || ebuf.ndim != 1){
            throw py::value_error("pars and energyArray must be 1D");
        }

        validate_par_size(NumPars, pbuf.size);

        if (ebuf.size < 2){
            throw py::value_error("Expected at least 2 bin edges");
        }

        const int nelem = ebuf.size - 1;

        // Can we easily zero out the arrays?
        auto result = py::array_t<double>(nelem);
        auto errors = std::vector<double>(nelem);

        BufferInfo obuf = result.request();

        double *pptr = static_cast<double *>(pbuf.ptr);
        double *eptr = static_cast<double *>(ebuf.ptr);
        double *optr = static_cast<double *>(obuf.ptr);

        model(eptr, nelem, pptr, spectrumNumber, optr, errors.data());
        return result;
    }


    template <typename T, xsccCall model, int NumPars>
    py::array_t<T> wrapper_con_C(
        const DoubleArray pars,
        const DoubleArray energyArray,
        DoubleArray inModel,
        const int spectrumNumber,
        const string initStr
    ) {
        BufferInfo pbuf = pars.request();
        BufferInfo ebuf = energyArray.request();
        BufferInfo mbuf = inModel.request();

        if (pbuf.ndim != 1 || ebuf.ndim != 1 || mbuf.ndim != 1) {
            throw py::value_error("pars and energyArray must be 1D");
        }

        validate_par_size(NumPars, pbuf.size);

        if (ebuf.size < 2) {
            throw py::value_error("Expected at least 2 bin edges");
        }

        validate_grid_size(ebuf.size, mbuf.size);

        // Should we force spectrumNumber >= 1?
        // We shouldn't be able to send in an invalid initStr so do not bother checking.

        const int nelem = ebuf.size - 1;

        // Can we easily zero out the arrays?
        auto result = py::array_t<double>(nelem);
        auto errors = std::vector<double>(nelem);

        BufferInfo obuf = result.request();

        double *pptr = static_cast<double *>(pbuf.ptr);
        double *eptr = static_cast<double *>(ebuf.ptr);
        double *mptr = static_cast<double *>(mbuf.ptr);
        double *optr = static_cast<double *>(obuf.ptr);

        std::copy(mptr, mptr + nelem, optr);

        model(eptr, nelem, pptr, spectrumNumber, optr, errors.data(), initStr.c_str());
        return result;
    }


    template <typename T, xsf77Call model, int NumPars>
    py::array_t<T> wrapper_con_f(
        const FloatArray pars,
        const FloatArray energyArray,
        FloatArray inModel,
        const int spectrumNumber
    ) {
        BufferInfo pbuf = pars.request();
        BufferInfo ebuf = energyArray.request();
        BufferInfo mbuf = inModel.request();

        if (pbuf.ndim != 1 || ebuf.ndim != 1 || mbuf.ndim != 1) {
            throw pybind11::value_error("pars and energyArray must be 1D");
        }

        validate_par_size(NumPars, pbuf.size);

        if (ebuf.size < 2) {
            throw pybind11::value_error("Expected at least 2 bin edges");
        }

        validate_grid_size(ebuf.size, mbuf.size);

        // Should we force spectrumNumber >= 1?
        // We shouldn't be able to send in an invalid initStr so do not bother checking.

        const int nelem = ebuf.size - 1;

        // Can we easily zero out the arrays?
        auto result = py::array_t<float>(nelem);
        auto errors = std::vector<float>(nelem);

        BufferInfo obuf = result.request();

        float *pptr = static_cast<float *>(pbuf.ptr);
        float *eptr = static_cast<float *>(ebuf.ptr);
        float *mptr = static_cast<float *>(mbuf.ptr);
        float *optr = static_cast<float *>(obuf.ptr);

        std::copy(mptr, mptr + nelem, optr);

        model(eptr, nelem, pptr, spectrumNumber, optr, errors.data());
        return result;
    }


    template <typename T>
    py::array_t<T> wrapper_table_model(
        const string filename,
        const string tableType,
	    FloatArray pars,
	    FloatArray energyArray,
	    const int spectrumNumber
    ) {
	    BufferInfo pbuf = pars.request();
	    BufferInfo ebuf = energyArray.request();

	    if (pbuf.ndim != 1 || ebuf.ndim != 1)
	      throw py::value_error("pars and energyArray must be 1D");

	    if (ebuf.size < 2)
	      throw py::value_error("Expected at least 2 bin edges");

	    // Should we force spectrumNumber >= 1?

	    const int nelem = ebuf.size - 1;

	    // Can we easily zero out the arrays?
	    auto result = py::array_t<float>(nelem);
	    auto errors = std::vector<float>(nelem);

	    BufferInfo obuf = result.request();

	    float *pptr = static_cast<float *>(pbuf.ptr);
	    float *eptr = static_cast<float *>(ebuf.ptr);
	    float *optr = static_cast<float *>(obuf.ptr);

	    tabint(
	        eptr, nelem, pptr, pbuf.size,
	        filename.c_str(), spectrumNumber,
		    tableType.c_str(), optr, errors.data()
		);
	    return result;
	  }


    template <xsccCall model, int NumPars>
    void wrapper_C_XLA_f32(void *out, void **in) {
        float *pptr = reinterpret_cast<float *>(in[0]);
        float *eptr = reinterpret_cast<float *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        const string initStr = "";  //*reinterpret_cast<string *>(in[5]);
        float *optr = reinterpret_cast<float *>(out);

        auto pars_ = std::vector<double>(NumPars);
        auto energyArray_ = std::vector<double>(nelem + 1);
        auto result_ = std::vector<double>(nelem);
        auto errors_ = std::vector<double>(nelem);

        double *pptr_ = pars_.data();
        double *eptr_ = energyArray_.data();
        double *optr_ = result_.data();
        float_to_double(eptr, eptr_, nelem + 1);

        for (int i = 0; i < batch; ++i) {
            float *pptr_i = pptr + i * NumPars;
            float *optr_i = optr + i * nelem;
            float_to_double(pptr_i, pptr_, NumPars);
            model(
                eptr_, nelem, pptr_, spectrumNumber, optr_,
                errors_.data(), initStr.c_str()
            );
            double_to_float(optr_, optr_i, nelem);
        }
    }

    template <xsccCall model, int NumPars>
    void wrapper_C_XLA_f64(void *out, void **in) {
        double *pptr = reinterpret_cast<double *>(in[0]);
        double *eptr = reinterpret_cast<double *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        const string initStr = "";  //*reinterpret_cast<string *>(in[5]);
        double *optr = reinterpret_cast<double *>(out);

        auto errors = std::vector<double>(nelem);

        for (int i = 0; i < batch; ++i) {
            double *pptr_i = pptr + i * NumPars;
            double *optr_i = optr + i * nelem;
            model(
                eptr, nelem, pptr_i, spectrumNumber, optr_i,
                errors.data(), initStr.c_str()
            );
        }
    }


    template <xsf77Call model, int NumPars>
    void wrapper_f_XLA_f32(void *out, void **in) {
        float *pptr = reinterpret_cast<float *>(in[0]);
        float *eptr = reinterpret_cast<float *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        float *optr = reinterpret_cast<float *>(out);

        auto errors = std::vector<float>(nelem);

        for (int i = 0; i < batch; ++i) {
            float *pptr_i = pptr + i * NumPars;
            float *optr_i = optr + i * nelem;
            model(
                eptr, nelem, pptr_i, spectrumNumber, optr_i, errors.data()
            );
        }
    }

    template <xsf77Call model, int NumPars>
    void wrapper_f_XLA_f64(void *out, void **in) {
        double *pptr = reinterpret_cast<double *>(in[0]);
        double *eptr = reinterpret_cast<double *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        double *optr = reinterpret_cast<double *>(out);

        auto pars_ = std::vector<float>(NumPars);
        auto energyArray_ = std::vector<float>(nelem + 1);
        auto result_ = std::vector<float>(nelem);
        auto errors_ = std::vector<float>(nelem);

        float *pptr_ = pars_.data();
        float *eptr_ = energyArray_.data();
        float *optr_ = result_.data();
        double_to_float(eptr, eptr_, nelem + 1);

        for (int i = 0; i < batch; ++i) {
            double *pptr_i = pptr + i * NumPars;
            double *optr_i = optr + i * nelem;
            double_to_float(pptr_i, pptr_, NumPars);
            model(
                eptr_, nelem, pptr_, spectrumNumber, optr_,
                errors_.data()
            );
            float_to_double(optr_, optr_i, nelem);
        }
        double_to_float(pptr, pptr_, NumPars);
    }


    template <xsF77Call model, int NumPars>
    void wrapper_F_XLA_f32(void *out, void **in) {
        float *pptr = reinterpret_cast<float *>(in[0]);
        float *eptr = reinterpret_cast<float *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        float *optr = reinterpret_cast<float *>(out);

        auto pars_ = std::vector<double>(NumPars);
        auto energyArray_ = std::vector<double>(nelem + 1);
        auto result_ = std::vector<double>(nelem);
        auto errors_ = std::vector<double>(nelem);

        double *pptr_ = pars_.data();
        double *eptr_ = energyArray_.data();
        double *optr_ = result_.data();
        float_to_double(eptr, eptr_, nelem + 1);

        for (int i = 0; i < batch; ++i) {
            float *pptr_i = pptr + i * NumPars;
            float *optr_i = optr + i * nelem;
            float_to_double(pptr_i, pptr_, NumPars);
            model(
                eptr_, nelem, pptr_, spectrumNumber, optr_, errors_.data()
            );
            double_to_float(optr_, optr_i, nelem);
        }
    }

    template <xsF77Call model, int NumPars>
    void wrapper_F_XLA_f64(void *out, void **in) {
        double *pptr = reinterpret_cast<double *>(in[0]);
        double *eptr = reinterpret_cast<double *>(in[1]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[2]);
        const int nelem = *reinterpret_cast<int *>(in[3]);
        const int batch = *reinterpret_cast<int *>(in[4]);
        double *optr = reinterpret_cast<double *>(out);

        auto errors = std::vector<double>(nelem);

        for (int i = 0; i < batch; ++i) {
            double *pptr_i = pptr + i * NumPars;
            double *optr_i = optr + i * nelem;
            model(
                eptr, nelem, pptr_i, spectrumNumber, optr_i,
                errors.data()
            );
        }
    }


    template <xsccCall model, int NumPars>
    void wrapper_con_C_XLA_f32(void *out, void **in) {
        float *pptr = reinterpret_cast<float *>(in[0]);
        float *eptr = reinterpret_cast<float *>(in[1]);
        float *mptr = reinterpret_cast<float *>(in[2]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[3]);
        const int nelem = *reinterpret_cast<int *>(in[4]);
        const int pbatch = *reinterpret_cast<int *>(in[5]);
        const int mbatch = *reinterpret_cast<int *>(in[6]);
        const string initStr = "";  //*reinterpret_cast<string *>(in[7]);
        float *optr = reinterpret_cast<float *>(out);

        int batch = std::max(1, std::max(pbatch, mbatch));
        auto params_ = std::vector<double>(NumPars * batch);
        auto model_ = std::vector<double>(nelem * batch);
        double *pptr_ = params_.data();
        double *mptr_ = model_.data();

        if (pbatch >= 1 && mbatch == 1) {
            float_to_double(pptr, pptr_, NumPars * batch);
            for (int i = 0; i < batch; ++i) {
                float_to_double(mptr, mptr_ + i * nelem, nelem);
            }
        } else if (pbatch == 1 && mbatch > 1) {
            float_to_double(mptr, mptr_, nelem * batch);
            for (int i = 0; i < batch; ++i) {
                float_to_double(pptr, pptr_ + i * NumPars, NumPars);
            }
        } else {
            if (pbatch > 1 && mbatch > 1 && pbatch == mbatch) {
                float_to_double(pptr, pptr_, NumPars * batch);
                float_to_double(mptr, mptr_, nelem * batch);
            } else {
                throw std::invalid_argument("Invalid batch sizes");
            }
        }

        auto energyArray_ = std::vector<double>(nelem + 1);
        auto errors_ = std::vector<double>(nelem);

        double *eptr_ = energyArray_.data();
        float_to_double(eptr, eptr_, nelem + 1);

        for (int i = 0; i < batch; ++i) {
            float *optr_i = optr + i * nelem;
            double *pptr_i = pptr_ + i * NumPars;
            double *mptr_i = mptr_ + i * nelem;
            model(
                eptr_, nelem, pptr_i, spectrumNumber, mptr_i,
                errors_.data(), initStr.c_str()
            );
            double_to_float(mptr_i, optr_i, nelem);
        }
    }

    template <xsccCall model, int NumPars>
    void wrapper_con_C_XLA_f64(void *out, void **in) {
        double *pptr = reinterpret_cast<double *>(in[0]);
        double *eptr = reinterpret_cast<double *>(in[1]);
        double *mptr = reinterpret_cast<double *>(in[2]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[3]);
        const int nelem = *reinterpret_cast<int *>(in[4]);
        const int pbatch = *reinterpret_cast<int *>(in[5]);
        const int mbatch = *reinterpret_cast<int *>(in[6]);
        const string initStr = "";  //*reinterpret_cast<string *>(in[7]);
        double *optr = reinterpret_cast<double *>(out);

        int batch = std::max(1, std::max(pbatch, mbatch));
        auto params_ = std::vector<double>(NumPars * batch);
        auto model_ = std::vector<double>(nelem * batch);
        double *pptr_ = params_.data();
        double *mptr_ = model_.data();

        if (pbatch >= 1 && mbatch == 1) {
            std::copy(pptr, pptr + NumPars * batch, pptr_);
            for (int i = 0; i < batch; ++i) {
                std::copy(mptr, mptr + nelem, mptr_ + i * nelem);
            }
        } else if (pbatch == 1 && mbatch > 1) {
            std::copy(mptr, mptr + nelem * batch, mptr_);
            for (int i = 0; i < batch; ++i) {
                std::copy(pptr, pptr + NumPars, pptr_ + i * NumPars);
            }
        } else {
            if (pbatch > 1 && mbatch > 1 && pbatch == mbatch) {
                std::copy(pptr, pptr + NumPars * batch, pptr_);
                std::copy(mptr, mptr + nelem * batch, mptr_);
            } else {
                throw std::invalid_argument("Invalid batch sizes");
            }
        }

        auto errors_ = std::vector<double>(nelem);

        for (int i = 0; i < batch; ++i) {
            double *pptr_i = pptr_ + i * NumPars;
            double *mptr_i = mptr_ + i * nelem;
            double *optr_i = optr + i * nelem;
            std::copy(mptr_i, mptr_i + nelem, optr_i);
            model(
                eptr, nelem, pptr_i, spectrumNumber, optr_i,
                errors_.data(), initStr.c_str()
            );
        }
    }


    template <xsf77Call model, int NumPars>
    void wrapper_con_f_XLA_f32(void *out, void **in) {
        float *pptr = reinterpret_cast<float *>(in[0]);
        float *eptr = reinterpret_cast<float *>(in[1]);
        float *mptr = reinterpret_cast<float *>(in[2]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[3]);
        const int nelem = *reinterpret_cast<int *>(in[4]);
        const int pbatch = *reinterpret_cast<int *>(in[5]);
        const int mbatch = *reinterpret_cast<int *>(in[6]);
        float *optr = reinterpret_cast<float *>(out);

        int batch = std::max(1, std::max(pbatch, mbatch));
        auto params_ = std::vector<float>(NumPars * batch);
        auto model_ = std::vector<float>(nelem * batch);
        float *pptr_ = params_.data();
        float *mptr_ = model_.data();

        if (pbatch >= 1 && mbatch == 1) {
            std::copy(pptr, pptr + NumPars * batch, pptr_);
            for (int i = 0; i < batch; ++i) {
                std::copy(mptr, mptr + nelem, mptr_ + i * nelem);
            }
        } else if (pbatch == 1 && mbatch > 1) {
            std::copy(mptr, mptr + nelem * batch, mptr_);
            for (int i = 0; i < batch; ++i) {
                std::copy(pptr, pptr + NumPars, pptr_ + i * NumPars);
            }
        } else {
            if (pbatch > 1 && mbatch > 1 && pbatch == mbatch) {
                std::copy(pptr, pptr + NumPars * batch, pptr_);
                std::copy(mptr, mptr + nelem * batch, mptr_);
            } else {
                throw std::invalid_argument("Invalid batch sizes");
            }
        }

        auto errors_ = std::vector<float>(nelem);

        for (int i = 0; i < batch; ++i) {
            float *pptr_i = pptr + i * NumPars;
            float *mptr_i = mptr_ + i * nelem;
            float *optr_i = optr + i * nelem;
            std::copy(mptr_i, mptr_i + nelem, optr_i);
            model(eptr, nelem, pptr_i, spectrumNumber, optr_i, errors_.data());
        }
    }

    template <xsf77Call model, int NumPars>
    void wrapper_con_f_XLA_f64(void *out, void **in) {
        double *pptr = reinterpret_cast<double *>(in[0]);
        double *eptr = reinterpret_cast<double *>(in[1]);
        double *mptr = reinterpret_cast<double *>(in[2]);
        const int spectrumNumber = *reinterpret_cast<int *>(in[3]);
        const int nelem = *reinterpret_cast<int *>(in[4]);
        const int pbatch = *reinterpret_cast<int *>(in[5]);
        const int mbatch = *reinterpret_cast<int *>(in[6]);
        double *optr = reinterpret_cast<double *>(out);

        int batch = std::max(1, std::max(pbatch, mbatch));
        auto params_ = std::vector<float>(NumPars * batch);
        auto model_ = std::vector<float>(nelem * batch);
        float *pptr_ = params_.data();
        float *mptr_ = model_.data();

        if (pbatch >= 1 && mbatch == 1) {
            double_to_float(pptr, pptr_, NumPars * batch);
            for (int i = 0; i < batch; ++i) {
                double_to_float(mptr, mptr_ + i * nelem, nelem);
            }
        } else if (pbatch == 1 && mbatch > 1) {
            double_to_float(mptr, mptr_, nelem * batch);
            for (int i = 0; i < batch; ++i) {
                double_to_float(pptr, pptr_ + i * NumPars, NumPars);
            }
        } else {
            if (pbatch > 1 && mbatch > 1 && pbatch == mbatch) {
                double_to_float(pptr, pptr_, NumPars * batch);
                double_to_float(mptr, mptr_, nelem * batch);
            } else {
                throw std::invalid_argument("Invalid batch sizes");
            }
        }

        auto energyArray_ = std::vector<float>(nelem + 1);
        auto errors_ = std::vector<float>(nelem);

        float *eptr_ = energyArray_.data();
        double_to_float(eptr, eptr_, nelem + 1);

        for (int i = 0; i < batch; ++i) {
            double *optr_i = optr + i * nelem;
            float *pptr_i = pptr_ + i * NumPars;
            float *mptr_i = mptr_ + i * nelem;
            model(
                eptr_, nelem, pptr_i, spectrumNumber, mptr_i,
                errors_.data()
            );
            float_to_double(mptr_i, optr_i, nelem);
        }
    }


    template <typename T>
    py::capsule EncapsulateFunction(T *fn) {
        return py::capsule((void *)fn, "xla._CUSTOM_CALL_TARGET");
    }
}

#endif
