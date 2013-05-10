/*
    Nickolas Fotopoulos (nvf@gravity.phys.uwm.edu)
    Fr library wrapper

    Functions: frgetvect, frputvect

    See individual docstrings for more information.

    Requires: numpy, FrameL
*/

#define OOM_ERROR printf("Unable to allocate space for data.\n");return PyErr_NoMemory();
#define CHECK_ERROR if (PyErr_Occurred()) {Py_XDECREF(framedict); Py_DECREF(channellist_iter); FrameFree(frame); return NULL;}
#define MAX_VECT_DIMS 10
#define MAX_STR_LEN 256

#include <Python.h>
#include <numpy/arrayobject.h>
#include <FrameL.h>

#if PY_VERSION_HEX < 0x02040000
#define Py_RETURN_NONE return Py_INCREF(Py_None), Py_None
#define Py_RETURN_TRUE return Py_INCREF(Py_True), Py_True
#define Py_RETURN_FALSE return Py_INCREF(Py_False), Py_False
#endif

static PyObject *author;
static PyObject *PyExc_FrError;

const char FrDocstring[] =
"    Nickolas Fotopoulos (nvf@gravity.phys.uwm.edu)\n"
"    Fr library wrapper\n"
"\n"
"    Functions: frgetvect, frgetvect1d, frputvect\n"
"    See individual docstrings for more information.\n"
"\n"
"    Requires: numpy (>=1.0), FrameL\n";

/* Some helper functions */
/* The PyDict_ExtractX functions will extract objects of a certain type from
a dict and convert them into C datatypes.  They should perform type checking
and raise appropriate exceptions. */

void PyDict_ExtractString(char out[MAX_STR_LEN], PyObject *dict, char *key) {
    char msg[MAX_STR_LEN];
    PyObject *temp;

    temp = PyDict_GetItemString(dict, key);

    if (temp == NULL) {
        snprintf(msg, MAX_STR_LEN, "%s not in dict", key);
        PyErr_SetString(PyExc_KeyError, msg);
        return;
    } else if (!PyString_Check(temp)) {
        snprintf(msg, MAX_STR_LEN, "%s is not a string", key);
        PyErr_SetString(PyExc_TypeError, msg);
        return;
    }
    strncpy(out, PyString_AsString(temp), MAX_STR_LEN);
    return;
}

double PyDict_ExtractDouble(PyObject *dict, char *key) {
    char msg[MAX_STR_LEN];
    double ret;
    PyObject *temp;

    temp = PyDict_GetItemString(dict, key);

    if (temp == NULL) {
        snprintf(msg, MAX_STR_LEN, "%s not in dict", key);
        PyErr_SetString(PyExc_KeyError, msg);
        return -1.;
    } else if (!PyNumber_Check(temp)) {
        snprintf(msg, MAX_STR_LEN, "%s is not a number", key);
        PyErr_SetString(PyExc_KeyError, msg);
        return -1.;
    }
    ret = PyFloat_AsDouble(temp);  // autocasts to PyFloat from others
    return ret;
}

/* The main functions */

const char frgetvectdocstring[] =
"frgetvect(filename, channel, start=-1, span=-1, verbose=False)\n"
"\n"
"Python adaptation of frgetvect (based on Matlab frgetvect).\n"
"Reads a vector from a Fr file to a numpy array.\n"
"\n"
"The input arguments are:\n"
"   1) filename - accepts name of frame file or FFL.\n"
"   2) channel - channel name (ADC, SIM, or PROC channel)\n"
"   3) start - starting GPS time (default = -1)\n"
"              A value <=0 will read the file's start time from the TOC.\n"
"   4) span - span of data in seconds (default = -1)\n"
"             A value <=0 will compute the file's span from the TOC.\n"
"   5) verbose - Verbose (True) or silent (False) (default = False)\n"
"\n"
"Returned data (in a tuple):\n"
"   1) Vector data as a numpy array\n"
"   2) GPS start time\n"
"   3) x-axes start values as a tuple of floats (for time-series, this is\n"
"      generally an offset from the GPS start time)\n"
"   4) x-axes spacings as a tuple of floats\n"
"   5) Units of x-axes as a tuple of strings\n"
"   6) Unit of y-axis as a string\n";

static PyObject *frgetvect(PyObject *self, PyObject *args, PyObject *keywds) {

    FrFile *iFile;
    FrVect *vect;
    int verbose, i, nDim;
    long nData;
    double start, span;
    char *filename, *channel, msg[MAX_STR_LEN];

    npy_intp shape[MAX_VECT_DIMS];

    npy_int16 *data_int16;
    npy_int32 *data_int32;
    npy_int64 *data_int64;
    npy_uint8 *data_uint8;
    npy_uint16 *data_uint16;
    npy_uint32 *data_uint32;
    npy_uint64 *data_uint64;
    npy_float32 *data_float32;
    npy_float64 *data_float64;

    static char *kwlist[] = {"filename", "channel", "start", "span",
                             "verbose", NULL};

    PyObject *out1, *out2, *out3, *out4, *out5, *out6;

    /*--------------- unpack arguments --------------------*/
    start = -1.;
    span = -1.;
    verbose = 0;

    /* The | in the format string indicates the next arguments are
       optional.  They are simply not assigned anything. */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "ss|ddi", kwlist,
        &filename, &channel, &start, &span, &verbose)) {
        PyErr_SetNone(PyExc_ValueError);
        return NULL;
    }

    FrLibSetLvl(verbose);

    /*-------------- open file --------------------------*/

    if (verbose > 0) {
        printf("Opening %s for channel %s (start=%.2f, span=%.2f).\n",
            filename, channel, start, span);
    }

    iFile = FrFileINew(filename);
    if (iFile == NULL){
        snprintf(msg, MAX_STR_LEN, "%s", FrErrorGetHistory());
        PyErr_SetString(PyExc_IOError, msg);
        return NULL;
    }

    if (verbose > 0){
        printf("Opened %s.\n", filename);
    }

    // Start and span auto-detection requires a TOC in the frame file.
    if (start == -1.) {
        start = FrFileITStart(iFile);
    }
    if (span == -1.) {
        span = FrFileITEnd(iFile) - start;
    }

    /*-------------- get vector --------------------------*/
    vect = FrFileIGetVect(iFile, channel, start, span);

    if (verbose > 0) FrVectDump(vect, stdout, verbose);
    if (vect == NULL) {
        /* Try to open it as StaticData */
        FrStatData *sd;
        /* Here I'd like to do
         *   sd = FrStatDataReadT(iFile, channel, start);
         * but FrStatDataReadT does *not* return samples after
         * "start". Doh. Instead, I have to do this:
         */
        double frstart = FrFileITStart(iFile);
        sd = FrStatDataReadT(iFile, channel, frstart);
        /* and more below */
        if (verbose > 0) FrStatDataDump(sd, stdout, verbose);
        if (sd == NULL) {
            snprintf(msg, MAX_STR_LEN, "In file %s, vector not found: %s", filename, channel);
            FrFileIEnd(iFile);
            PyErr_SetString(PyExc_KeyError, msg);
            return NULL;
        }
        if (sd->next != NULL) {
            snprintf(msg, MAX_STR_LEN, "In file %s, staticData channel %s has next!=NULL. "
                    "Freaking out.", filename, channel);
            FrFileIEnd(iFile);
            PyErr_SetString(PyExc_KeyError, msg);
            return NULL;
        }
        vect = sd->data;
        if (vect == NULL) {
            snprintf(msg, MAX_STR_LEN, "In file %s, staticData channel %s has no vector. "
                    "Freaking out.", filename, channel);
            FrFileIEnd(iFile);
            PyErr_SetString(PyExc_KeyError, msg);
            return NULL;
        }
        if (vect->nDim != 1) {
            snprintf(msg, MAX_STR_LEN, "In file %s, staticData channel %s has multiple "
                    "dimensions. Freaking out.", filename, channel);
            FrFileIEnd(iFile);
            PyErr_SetString(PyExc_KeyError, msg);
            return NULL;
        }

        /* Recompute limits and pointers, so "vect" contains only the
         * subset of data we requested */
        if (vect->nData > span / vect->dx[0]) {
            vect->nx[0] = span / vect->dx[0];
            vect->nData = vect->nx[0];
        }
        if (frstart < start) {  /* thank you FrStatDataReadT() */
            int shift = (start - frstart) / vect->dx[0];
            if      (vect->type == FR_VECT_2S)   vect->dataS  += shift;
            else if (vect->type == FR_VECT_4S)   vect->dataI  += shift;
            else if (vect->type == FR_VECT_8S)   vect->dataL  += shift;
            else if (vect->type == FR_VECT_1U)   vect->dataU  += shift;
            else if (vect->type == FR_VECT_2U)   vect->dataUS += shift;
            else if (vect->type == FR_VECT_4U)   vect->dataUI += shift;
            else if (vect->type == FR_VECT_8U)   vect->dataUL += shift;
            else if (vect->type == FR_VECT_4R)   vect->dataF  += shift;
            else if (vect->type == FR_VECT_8R)   vect->dataD  += shift;
            // Note the 2* shift for complex types
            else if (vect->type == FR_VECT_8C)   vect->dataF  += 2 * shift;
            else if (vect->type == FR_VECT_16C)  vect->dataD  += 2 * shift;
            // If none of these types, it will fail later
        }
    }

    if (verbose > 0){
        printf("Extracted channel %s successfully!\n", channel);
    }

    nData = vect->nData;
    nDim = vect->nDim;

    /*-------- copy data ------*/

    for (i=0; i<nDim; i++) {
        shape[i] = (npy_intp)vect->nx[i];
    }

    // Both FrVect and Numpy store data in C array order (vs Fortran)
    if(vect->type == FR_VECT_2S){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_INT16);
        data_int16 = (npy_int16 *)PyArray_DATA(out1);
        if (data_int16==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_int16[i] = vect->dataS[i];}}
    else if(vect->type == FR_VECT_4S){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_INT32);
        data_int32 = (npy_int32 *)PyArray_DATA(out1);
        if (data_int32==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_int32[i] = vect->dataI[i];}}
    else if(vect->type == FR_VECT_8S){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_INT64);
        data_int64 = (npy_int64 *)PyArray_DATA(out1);
        if (data_int64==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_int64[i] = vect->dataL[i];}}
    else if(vect->type == FR_VECT_1U){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_UINT8);
        data_uint8 = (npy_uint8 *)PyArray_DATA(out1);
        if (data_uint8==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_uint8[i] = vect->dataU[i];}}
    else if(vect->type == FR_VECT_2U){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_UINT16);
        data_uint16 = (npy_uint16 *)PyArray_DATA(out1);
        if (data_uint16==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_uint16[i] = vect->dataUS[i];}}
    else if(vect->type == FR_VECT_4U){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_UINT32);
        data_uint32 = (npy_uint32 *)PyArray_DATA(out1);
        if (data_uint32==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_uint32[i] = vect->dataUI[i];}}
    else if(vect->type == FR_VECT_8U){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_UINT64);
        data_uint64 = (npy_uint64 *)PyArray_DATA(out1);
        if (data_uint64==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_uint64[i] = vect->dataUL[i];}}
    else if(vect->type == FR_VECT_4R){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_FLOAT32);
        data_float32 = (npy_float32 *)PyArray_DATA(out1);
        if (data_float32==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_float32[i] = vect->dataF[i];}}
    else if(vect->type == FR_VECT_8R){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_FLOAT64);
        data_float64 = (npy_float64 *)PyArray_DATA(out1);
        if (data_float64==NULL) {OOM_ERROR;}
        for(i=0; i<nData; i++) {data_float64[i] = vect->dataD[i];}}
    // Note the 2*nData in the for loop for complex types
    else if(vect->type == FR_VECT_8C){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_COMPLEX64);
        data_float32 = (npy_float32 *)PyArray_DATA(out1);
        if (data_float32==NULL) {OOM_ERROR;}
        for(i=0; i<2*nData; i++) {data_float32[i] = vect->dataF[i];}}
    else if(vect->type == FR_VECT_16C){
        out1 = PyArray_SimpleNew(nDim, shape, NPY_COMPLEX128);
        data_float64 = (npy_float64 *)PyArray_DATA(out1);
        if (data_float64==NULL) {OOM_ERROR;}
        for(i=0; i<2*nData; i++) {data_float64[i] = vect->dataD[i];}}
    else{
        snprintf(msg, MAX_STR_LEN, "Unrecognized vect->type (= %d)\n", vect->type);
        FrVectFree(vect);
        FrFileIEnd(iFile);
        PyErr_SetString(PyExc_TypeError, msg);
        return NULL;
    }

    /*------------- other outputs ------------------------*/
    // output2 = gps start time
    out2 = PyFloat_FromDouble(vect->GTime);

    // output3 = x-axes start values as a tuple of PyFloats
    // output4 = x-axes spacings as a tuple of PyFloats
    // output5 = x-axes units as a tuple of strings
    out3 = PyTuple_New((Py_ssize_t)nDim);
    out4 = PyTuple_New((Py_ssize_t)nDim);
    out5 = PyTuple_New((Py_ssize_t)nDim);
    for (i=0; i<nDim; i++) {
        PyTuple_SetItem(out3, (Py_ssize_t)i, PyFloat_FromDouble(vect->startX[i]));
        PyTuple_SetItem(out4, (Py_ssize_t)i, PyFloat_FromDouble(vect->dx[i]));
        PyTuple_SetItem(out5, (Py_ssize_t)i, PyString_FromString(vect->unitX[i]));
    }

    // output6 = unitY as a string
    out6 = PyString_FromString(vect->unitY);
    /*------------- clean up -----------------------------*/

    FrVectFree(vect);
    FrFileIEnd(iFile);
    return Py_BuildValue("(NNNNNN)",out1,out2,out3,out4,out5, out6);
};

const char frgetvect1ddocstring[] =
"frgetvect1d(filename, channel, start=-1, span=-1, verbose=False)\n"
"\n"
"1-D version of the multi-dimensional frgetvect.\n"
"Reads a one-dimensional vector from a Fr file to a numpy array.  Will raise\n"
"an exception if invoked on multi-dimensional data."
"\n"
"The input arguments are:\n"
"   1) filename - accepts name of frame file or FFL.\n"
"   2) channel - channel name (ADC, SIM or PROC channel)\n"
"   3) start - starting GPS time (default = -1)\n"
"              A value <=0 will read from the first available frame.\n"
"   4) span - span of data in seconds (default = -1)\n"
"             A value <=0 will return the entirety of the first vector with\n"
"             a matching channel name.  Defaulting span will force a default\n"
"             start.\n"
"   5) verbose - Verbose (True) or silent (False) (default = False)\n"
"\n"
"Returned data (in a tuple):\n"
"   1) Vector data as a numpy array\n"
"   2) GPS start time\n"
"   3) x-axis start value as a float (for time-series, this is\n"
"      generally an offset from the GPS start time)\n"
"   4) x-axis spacing as a float\n"
"   5) Unit of x-axis as a string\n"
"   6) Unit of y-axis as a string\n";

static PyObject *frgetvect1d(PyObject *self, PyObject *args, PyObject *keywds) {
    PyObject *out, *temp, *temp2;

    out = frgetvect(self, args, keywds);
    if (out == NULL) {
        return NULL;
    }

    // Check lengths and unpack outputs 3, 4, and 5.  Be paranoid about
    // output 3 only.
    temp = PyTuple_GetItem(out, (Py_ssize_t) 2);
    if (temp==NULL || PyTuple_Size(temp) != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "frgetvect1d invoked on a multi-dimensional vector");
        Py_DECREF(out);
        return NULL;
    }
    temp2 = PyTuple_GetItem(temp, (Py_ssize_t) 0);
    Py_INCREF(temp2);
    PyTuple_SetItem(out, (Py_ssize_t) 2, temp2); // Python DECREFs temp and temp2 here

    temp = PyTuple_GetItem(out, (Py_ssize_t) 3);
    temp2 = PyTuple_GetItem(temp, (Py_ssize_t) 0);
    Py_INCREF(temp2);
    PyTuple_SetItem(out, (Py_ssize_t) 3, temp2);

    temp = PyTuple_GetItem(out, (Py_ssize_t) 4);
    temp2 = PyTuple_GetItem(temp, (Py_ssize_t) 0);
    Py_INCREF(temp2);
    PyTuple_SetItem(out, (Py_ssize_t) 4, temp2);

    if (PyErr_Occurred()) {
        Py_DECREF(out);
        return NULL;
    }

    return out;
}

const char frputvectdocstring[] =
"frputvect(filename, channellist, history='', verbose=False)\n"
"\n"
"The inverse of frgetvect -- write numpy arrays to a Fr frame file.\n"
"\n"
"The input arguments are:\n"
"    1) filename - name of file to write.\n"
"    2) channellist - list of dictionaries with the fields below:\n"
"        1) name - channel name - string\n"
"        2) data - list of one-dimensional vectors to write\n"
"        3) start - lower limit of the x-axis in GPS time or Hz\n"
"        4) dx - spacing of x-axis in seconds or Hz\n"
"        5) x_unit - unit of x-axis as a string (default = '')\n"
"        6) y_unit - unit of y-axis as a string (default = '')\n"
"        7) kind - 'PROC', 'ADC', or 'SIM' (default = 'PROC')\n"
"        8) type - type of data (default = 1):\n"
"               0 - Unknown/undefined\n"
"               1 - Time series\n"
"               2 - Frequency series\n"
"               3 - Other 1-D series\n"
"               4 - Time-frequency\n"
"               5 - Wavelets\n"
"               6 - Multi-dimensional\n"
"        9) subType - sub-type of frequency series (default = 0):\n"
"               0 - Unknown/undefined\n"
"               1 - DFT\n"
"               2 - Amplitude spectral density\n"
"               3 - Power spectral density\n"
"               4 - Cross spectral density\n"
"               5 - Coherence\n"
"               6 - Transfer function\n"
"   3) history - history string (default = '')\n"
"   4) verbose - Verbose (True) or silent (False) (default = False)\n"
"\n"
"Returns None";

static PyObject *frputvect(PyObject *self, PyObject *args, PyObject *keywds) {
    FrFile *oFile;
    FrameH *frame;
    FrProcData *proc;
    FrAdcData *adc;
    FrSimData *sim;
    FrVect *vect;
    int verbose=0, nData, nBits, type, subType, arrayType;
    double dx, sampleRate, start;
    char blank[] = "";
    char *filename=NULL, *history=NULL;
    char channel[MAX_STR_LEN], x_unit[MAX_STR_LEN], y_unit[MAX_STR_LEN], kind[MAX_STR_LEN];
    PyObject *temp;
    char msg[MAX_STR_LEN];

    PyObject *channellist, *channellist_iter, *framedict, *array;
    PyArrayIterObject *arrayIter;
    PyArray_Descr *temp_descr;

    static char *kwlist[] = {"filename", "channellist", "history", "verbose",
                             NULL};

    /*--------------- unpack arguments --------------------*/
    verbose = 0;

    /* The | in the format string indicates the next arguments are
       optional.  They are simply not assigned anything. */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "sO|si", kwlist,
        &filename, &channellist, &history, &verbose)) {
        Py_RETURN_NONE;
    }

    FrLibSetLvl(verbose);

    if (history == NULL) {
        history = blank;
    }

    /*-------- create frames, create vectors, and fill them. ------*/

    // Channel-list must be any type of sequence
    if (!PySequence_Check(channellist)) {
        PyErr_SetNone(PyExc_TypeError);
        return NULL;
    }

    // Get channel name from first dictionary
    framedict = PySequence_GetItem(channellist, (Py_ssize_t)0);
    if (framedict == NULL) {
        PyErr_SetString(PyExc_ValueError, "channellist is empty!");
        return NULL;
    }

    PyDict_ExtractString(channel, framedict, "name");
    Py_XDECREF(framedict);
    if (PyErr_Occurred()) {return NULL;}

    if (verbose > 0) {
        printf("Creating frame %s...\n", channel);
    }

    frame = FrameNew(channel);
    if (frame == NULL) {
        snprintf(msg, MAX_STR_LEN, "FrameNew failed (%s)", FrErrorGetHistory());
        PyErr_SetString(PyExc_FrError, msg);
        return NULL;
    }

    if (verbose > 0) {
        printf("Now iterating...\n");
    }

    // Iterators allow one to deal with non-contiguous arrays
    channellist_iter = PyObject_GetIter(channellist);
    arrayIter = NULL;
    while ((framedict = PyIter_Next(channellist_iter))) {
        if (verbose > 0) {
            printf("In loop...\n");
        }

        // Extract quantities from dict -- all borrowed references
        PyDict_ExtractString(channel, framedict, "name");
        CHECK_ERROR;

        start = PyDict_ExtractDouble(framedict, "start");
        CHECK_ERROR;

        dx = PyDict_ExtractDouble(framedict, "dx");
        CHECK_ERROR;

        array = PyDict_GetItemString(framedict, "data");
        if (!PyArray_Check(array)) {
            snprintf(msg, MAX_STR_LEN, "data is not an array");
            PyErr_SetString(PyExc_TypeError, msg);
        }
        CHECK_ERROR;

        nData = PyArray_SIZE(array);
        nBits = PyArray_ITEMSIZE(array);
        arrayType = PyArray_TYPE(array);

        // kind, x_unit, y_unit, type, and subType have default values
        temp = PyDict_GetItemString(framedict, "kind");
        if (temp != NULL) {strncpy(kind, PyString_AsString(temp), MAX_STR_LEN);}
        else {snprintf(kind, MAX_STR_LEN, "PROC");}

        temp = PyDict_GetItemString(framedict, "x_unit");
        if (temp != NULL) {strncpy(x_unit, PyString_AsString(temp), MAX_STR_LEN);}
        else {strncpy(x_unit, blank, MAX_STR_LEN);}

        temp = PyDict_GetItemString(framedict, "y_unit");
        if (temp != NULL) {strncpy(y_unit, PyString_AsString(temp), MAX_STR_LEN);}
        else {strncpy(y_unit, blank, MAX_STR_LEN);}

        temp = PyDict_GetItemString(framedict, "type");
        if (temp != NULL) {type = (int)PyInt_AsLong(temp);}
        else {type = 1;}

        temp = PyDict_GetItemString(framedict, "subType");
        if (temp != NULL) {subType = (int)PyInt_AsLong(temp);}
        else {subType = 0;}

        // check for errors
        CHECK_ERROR;
        if (dx <= 0 || array == NULL || nData==0) {
            temp = PyObject_Str(framedict);
            snprintf(msg, MAX_STR_LEN, "Input dictionary contents: %s", PyString_AsString(temp));
            Py_XDECREF(temp);
            FrameFree(frame);
            Py_XDECREF(framedict);
            Py_XDECREF(channellist_iter);
            PyErr_SetString(PyExc_ValueError, msg);
            return NULL;
        }


        if (verbose > 0) {
            printf("type = %d, subType = %d, start = %f, dx = %f\n",
                type, subType, start, dx);
        }

        sampleRate = 1./dx;

        if (verbose > 0) {
            printf("Now copying data to vector...\n");
        }

        // Create empty vector (-typecode ==> empty) with metadata,
        // then copy data to vector
        vect = NULL;
        arrayIter = (PyArrayIterObject *)PyArray_IterNew(array);
        if(arrayType == NPY_INT16) {
            vect = FrVectNew1D(channel,-FR_VECT_2S,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataS[arrayIter->index] = *((npy_int16 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_INT32) {
            vect = FrVectNew1D(channel,-FR_VECT_4S,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataI[arrayIter->index] = *((npy_int32 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_INT64) {
            vect = FrVectNew1D(channel,-FR_VECT_8S,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataL[arrayIter->index] = *((npy_int64 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_UINT8) {
            vect = FrVectNew1D(channel,-FR_VECT_1U,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataU[arrayIter->index] = *((npy_uint8 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_UINT16) {
            vect = FrVectNew1D(channel,-FR_VECT_2U,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataUS[arrayIter->index] = *((npy_uint16 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_UINT32) {
            vect = FrVectNew1D(channel,-FR_VECT_4U,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataUI[arrayIter->index] = *((npy_uint32 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_UINT64) {
            vect = FrVectNew1D(channel,-FR_VECT_8U,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataUL[arrayIter->index] = *((npy_uint64 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_FLOAT32) {
            vect = FrVectNew1D(channel,-FR_VECT_4R,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataF[arrayIter->index] = *((npy_float32 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        else if(arrayType == NPY_FLOAT64) {
            vect = FrVectNew1D(channel,-FR_VECT_8R,nData,dx,x_unit,y_unit);
            while (arrayIter->index < arrayIter->size) {
                vect->dataD[arrayIter->index] = *((npy_float64 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}}
        /* FrVects don't have complex pointers.  Numpy stores complex
           numbers in the same way, but we have to trick it into giving
           us a (real) float pointer. */
        else if(arrayType == NPY_COMPLEX64) {
            vect = FrVectNew1D(channel,-FR_VECT_8C,nData,dx,x_unit,y_unit);
            temp_descr = PyArray_DescrFromType(NPY_FLOAT32);
            temp = PyArray_View((PyArrayObject *)array, temp_descr, NULL);
            Py_XDECREF(temp_descr);
            Py_XDECREF(arrayIter);
            arrayIter = (PyArrayIterObject *)PyArray_IterNew(temp);
            while (arrayIter->index < arrayIter->size) {
                vect->dataF[arrayIter->index] = *((npy_float32 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}
            Py_XDECREF(temp);}
        else if(arrayType == NPY_COMPLEX128) {
            vect = FrVectNew1D(channel,-FR_VECT_16C,nData,dx,x_unit,y_unit);
            temp_descr = PyArray_DescrFromType(NPY_FLOAT64);
            temp = PyArray_View((PyArrayObject *)array, temp_descr, NULL);
            Py_XDECREF(temp_descr);
            Py_XDECREF(arrayIter);
            arrayIter = (PyArrayIterObject *)PyArray_IterNew(temp);
            while (arrayIter->index < arrayIter->size) {
                vect->dataD[arrayIter->index] = *((npy_float64 *)arrayIter->dataptr);
                PyArray_ITER_NEXT(arrayIter);}
            Py_XDECREF(temp);}
        else PyErr_SetString(PyExc_TypeError, msg);

        if (PyErr_Occurred()) {
            if (vect != NULL) FrVectFree(vect);
            FrameFree(frame);
            Py_XDECREF(framedict);
            Py_XDECREF(channellist_iter);
            Py_XDECREF(arrayIter);
            return NULL;
        }

        if (verbose > 0) {
            printf("Done copying...\n");
            FrameDump(frame, stdout, 6);
        }

        // Add Fr*Data to frame and attach vector to Fr*Data
        if (strncmp(kind, "PROC", MAX_STR_LEN)==0) {
            proc = FrProcDataNew(frame, channel, sampleRate, 1, nBits);
            FrVectFree(proc->data);
            proc->data = vect;
            proc->type = type;
            proc->subType = subType;
            frame->GTimeS = (npy_uint32)start;
            frame->GTimeN = (npy_uint32)((start-(frame->GTimeS))*1e9);
            if (type==1) {  // time series
                proc->tRange = nData*dx;
                frame->dt = nData*dx;
            } else if (type==2) {  // frequency series
                proc->fRange = nData*dx;
            }
        } else if (strncmp(kind, "ADC", MAX_STR_LEN)==0) {
            adc = FrAdcDataNew(frame, channel, sampleRate, 1, nBits);
            FrVectFree(adc->data);
            adc->data = vect;
            frame->dt = nData*dx;
            frame->GTimeS = (npy_uint32)start;
            frame->GTimeN = (npy_uint32)((start-(frame->GTimeS))*1e9);
        } else {// Already tested that kind is one of these strings above
            sim = FrSimDataNew(frame, channel, sampleRate, 1, nBits);
            FrVectFree(sim->data);
            sim->data = vect;
            frame->dt = nData*dx;
            frame->GTimeS = (npy_uint32)start;
            frame->GTimeN = (npy_uint32)((start-(frame->GTimeS))*1e9);
        }

        if (verbose > 0) {
            printf("Attached vect to frame.\n");
        }

        // Clean up (all python objects in loop should be borrowed references)
        Py_XDECREF(framedict);
        Py_XDECREF(arrayIter);
    } // end iteration over channellist

    Py_XDECREF(channellist_iter);
    // At this point, there should be no Python references left!

    /*------------- Write file -----------------------------*/
    oFile = FrFileONewH(filename, 1, history); // 1 ==> gzip contents

    if (oFile == NULL) {
        snprintf(msg, MAX_STR_LEN, "%s\n", FrErrorGetHistory());
        PyErr_SetString(PyExc_FrError, msg);
        FrFileOEnd(oFile);
        return NULL;
    }
    if (FrameWrite(frame, oFile) != FR_OK) {
        snprintf(msg, MAX_STR_LEN, "%s\n", FrErrorGetHistory());
        PyErr_SetString(PyExc_FrError, msg);
        FrFileOEnd(oFile);
        return NULL;
    }

    /* The FrFile owns data and vector memory. Do not free them separately. */
    FrFileOEnd(oFile);
    FrameFree(frame);
    Py_RETURN_NONE;
};

/*
 * Utility function to extract an FrEvent's parameters into a Python dictionary
 */
static PyObject *extract_event_dict(FrEvent *event) {
    PyObject *event_dict = NULL;
    size_t j;

    /* each FrEvent will be stored in a dict */
    event_dict = PyDict_New();
    if (!event_dict) return NULL;

    /* guarantee these parameters exist */
    PyDict_SetItemString(event_dict, "name",
        PyString_FromString(event->name));
    PyDict_SetItemString(event_dict, "comment",
        PyString_FromString(event->comment));
    PyDict_SetItemString(event_dict, "inputs",
        PyString_FromString(event->inputs));
    PyDict_SetItemString(event_dict, "GTimeS",
        PyLong_FromUnsignedLong(event->GTimeS));
    PyDict_SetItemString(event_dict, "GTimeN",
        PyLong_FromUnsignedLong(event->GTimeN));
    PyDict_SetItemString(event_dict, "timeBefore",
        PyFloat_FromDouble(event->timeBefore));
    PyDict_SetItemString(event_dict, "timeAfter",
        PyFloat_FromDouble(event->timeAfter));
    PyDict_SetItemString(event_dict, "eventStatus",
        PyLong_FromUnsignedLong(event->eventStatus));
    PyDict_SetItemString(event_dict, "amplitude",
        PyFloat_FromDouble(event->amplitude));
    PyDict_SetItemString(event_dict, "probability",
        PyFloat_FromDouble(event->probability));
    PyDict_SetItemString(event_dict, "statistics",
        PyString_FromString(event->statistics));

    /* additional parameters */
    for (j = 0; j < event->nParam; j++) {
        PyDict_SetItem(event_dict,
            PyString_FromString(event->parameterNames[j]),
            PyFloat_FromDouble(event->parameters[j]));
    }

    return event_dict;
}

const char frgeteventdocstring[] =
"frgetevent(filename, verbose=False)\n"
"\n"
"Extract the FrEvents from a given frame file.\n"
"\n"
"Returns a list of dicts, with each dict representing an FrEvent's fields "
"and values.\n";

static PyObject *frgetevent(PyObject *self, PyObject *args, PyObject *keywds) {
    FrFile *iFile=NULL;
    FrameH *frame=NULL;
    FrEvent *event=NULL;

    int verbose=0, status;
    char *filename=NULL;
    char msg[MAX_STR_LEN];

    PyObject *event_list=NULL, *event_dict=NULL, *verbose_obj=NULL,
        *output=NULL;

    static char *kwlist[] = {"filename", "verbose", NULL};

    /*--------------- unpack arguments --------------------*/
    /* The | in the format string indicates the next arguments are
       optional.  They are simply not assigned anything. */
    if (!PyArg_ParseTupleAndKeywords(args, keywds, "s|O", kwlist,
        &filename, &verbose_obj)) {
        Py_RETURN_NONE;
    }
    verbose = verbose_obj && PyObject_IsTrue(verbose_obj);
    FrLibSetLvl(verbose);

    /*-------------- set up file for reading --------------*/

    iFile = FrFileINew(filename);
    if (iFile == NULL){
        snprintf(msg, MAX_STR_LEN, "%s", FrErrorGetHistory());
        PyErr_SetString(PyExc_IOError, msg);
        return NULL;
    }

    /* require at least one frame in the file */
    frame = FrameRead(iFile);
    if (frame == NULL) {
        snprintf(msg, MAX_STR_LEN, "%s", FrErrorGetHistory());
        PyErr_SetString(PyExc_IOError, msg);
        FrFileIEnd(iFile);
        return NULL;
    }

    /*------ iterate, putting each event into output list ------*/

    event_list = PyList_New(0);
    if (!event_list) goto clean_C;
    do {
        for (event = frame->event; event != NULL; event = event->next) {
            event_dict = extract_event_dict(event);
            if (!event_dict) goto clean_C_and_Python;
            status = PyList_Append(event_list, event_dict);
            if (status == -1) {
                Py_DECREF(event_dict);
                goto clean_C_and_Python;
            }
        }
    } while ((frame = FrameRead(iFile)));

    /* error checking */
    /* NB: FrameL doesn't distinguish between EOF and an error */
    if (PyErr_Occurred()) goto clean_C_and_Python;

    /* if we've gotten here, we're all done; set the output to return */
    output = event_list;

    /*-------------- clean up and return --------------*/
    /*
       C always needs to be cleaned up, error or no.
       Python only needs to be cleaned up on a true error condition.
    */
    clean_C:
        FrameFree(frame);
        FrFileIEnd(iFile);
        return output;

    clean_C_and_Python:
        Py_XDECREF(event_list);
        FrameFree(frame);
        FrFileIEnd(iFile);
        return NULL;
}


/*
  Attach functions to module
*/

static PyMethodDef FrMethods[] = {
    {"frgetvect", (PyCFunction)frgetvect, METH_VARARGS|METH_KEYWORDS,
         frgetvectdocstring},
    {"frgetvect1d", (PyCFunction)frgetvect1d, METH_VARARGS|METH_KEYWORDS,
         frgetvect1ddocstring},
    {"frputvect", (PyCFunction)frputvect, METH_VARARGS|METH_KEYWORDS,
         frputvectdocstring},
    {"frgetevent", (PyCFunction)frgetevent, METH_VARARGS|METH_KEYWORDS,
         frgeteventdocstring},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initFr(void) {
    PyObject *m;
    m = Py_InitModule3("pylal.Fr", FrMethods, FrDocstring);

    import_array();

    PyExc_FrError = PyErr_NewException("Fr.FrError", NULL, NULL);
    author = PyString_FromString("Nickolas Fotopoulos <nvf@gravity.phys.uwm.edu>");

    Py_INCREF(PyExc_FrError);  // Recommended by extending & embedding doc
    PyModule_AddObject(m, "FrError", PyExc_FrError);
    PyModule_AddObject(m, "__author__", author);
};
