/*
 *
 * Copyright (C) 2006  Kipp C. Cannon
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation; either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */


/*
 * ============================================================================
 *
 *                Segments Module Component --- infinity Class
 *
 * ============================================================================
 */


#include <Python.h>
#include <stdlib.h>


#include <segments.h>


/*
 * ============================================================================
 *
 *                               infinity Class
 *
 * ============================================================================
 */


/*
 * Preallocated instances
 */


segments_Infinity *segments_PosInfinity;
segments_Infinity *segments_NegInfinity;


/*
 * Utilities
 */


static int segments_Infinity_Check(PyObject *obj)
{
	return obj ? PyObject_TypeCheck(obj, &segments_Infinity_Type) : 0;
}


/*
 * Methods
 */


static PyObject *__new__(PyTypeObject *subtype, PyObject *args, PyObject *kwds)
{
	int sign = +1;	/* return +infinity when called with no arg */
	PyObject *self;

	if(!PyArg_ParseTuple(args, "|i:infinity", &sign))
		return NULL;
	if(sign > 0)
		self = (PyObject *) segments_PosInfinity;
	else if (sign < 0)
		self = (PyObject *) segments_NegInfinity;
	else {
		PyErr_SetObject(PyExc_ValueError, args);
		return NULL;
	}

	Py_INCREF(self);
	return self;
}


static PyObject *__add__(PyObject *self, PyObject *other)
{
	if(segments_Infinity_Check(self)) {
		/* __add__ case */
		Py_INCREF(self);
		return self;
	} else if(segments_Infinity_Check(other)) {
		/* __radd__ case */
		Py_INCREF(other);
		return other;
	}
	PyErr_SetObject(PyExc_TypeError, self);
	return NULL;
}


static PyObject *__neg__(PyObject *self)
{
	PyObject *result;

	if(segments_Infinity_Check(self)) {
		if(self == (PyObject *) segments_PosInfinity)
			result = (PyObject *) segments_NegInfinity;
		else
			result = (PyObject *) segments_PosInfinity;
		Py_INCREF(result);
		return result;
	}
	PyErr_SetObject(PyExc_TypeError, self);
	return NULL;
}


static int __nonzero__(PyObject *self)
{
	if(segments_Infinity_Check(self))
		return 1;
	PyErr_SetObject(PyExc_TypeError, self);
	return -1;
}


static PyObject *__pos__(PyObject *self)
{
	if(segments_Infinity_Check(self)) {
		Py_INCREF(self);
		return self;
	}
	PyErr_SetObject(PyExc_TypeError, self);
	return NULL;
}


static PyObject *__repr__(PyObject *self)
{
	return PyString_FromString(self == (PyObject *) segments_PosInfinity ? "infinity" : "-infinity");
}


static PyObject *__reduce__(PyObject *self, PyObject *args)
{
	if(segments_Infinity_Check(self)) {
		Py_INCREF(&segments_Infinity_Type);
		return Py_BuildValue("(O,(i))", &segments_Infinity_Type, self == (PyObject *) segments_PosInfinity ? +1 : -1);
	}
	PyErr_SetObject(PyExc_TypeError, self);
	return NULL;
}


static PyObject *richcompare(PyObject *self, PyObject *other, int op_id)
{
	int s = segments_Infinity_Check(self) ? self == (PyObject *) segments_PosInfinity ? +1 : -1 : 0;
	int o = segments_Infinity_Check(other) ? other == (PyObject *) segments_PosInfinity ? +1 : -1 : 0;
	int d = s - o;
	PyObject *result;

	if(!(s || o)) {
		/* neither of the arguments is an Infinity instance */
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}

	switch(op_id) {
	case Py_LT:
		result = (d < 0) ? Py_True : Py_False;
		break;

	case Py_LE:
		result = (d <= 0) ? Py_True : Py_False;
		break;

	case Py_EQ:
		result = (d == 0) ? Py_True : Py_False;
		break;

	case Py_NE:
		result = (d != 0) ? Py_True : Py_False;
		break;

	case Py_GT:
		result = (d > 0) ? Py_True : Py_False;
		break;

	case Py_GE:
		result = (d >= 0) ? Py_True : Py_False;
		break;

	default:
		PyErr_BadInternalCall();
		return NULL;
	}

	Py_INCREF(result);
	return result;
}


static PyObject *__sub__(PyObject *self, PyObject *other)
{
	PyObject *result;

	if(segments_Infinity_Check(self))
		/* __sub__ case */
		result = self;
	else if(segments_Infinity_Check(other)) {
		/* __rsub__ case */
		if(other == (PyObject *) segments_PosInfinity)
			result = (PyObject *) segments_NegInfinity;
		else
			result = (PyObject *) segments_PosInfinity;
	} else {
		PyErr_SetObject(PyExc_TypeError, self);
		return NULL;
	}
	Py_INCREF(result);
	return result;
}


/*
 * Type information
 */


static PyNumberMethods as_number = {
	.nb_add = __add__,
	.nb_negative = __neg__,
	.nb_nonzero = __nonzero__,
	.nb_positive = __pos__,
	.nb_subtract = __sub__,
};


static struct PyMethodDef methods[] = {
	{"__reduce__", __reduce__, METH_NOARGS, "Pickle support"},
	{NULL,}
};


PyTypeObject segments_Infinity_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_as_number = &as_number,
	.tp_basicsize = sizeof(segments_Infinity),
	.tp_doc =
"The infinity object possess the algebraic properties necessary for\n" \
"use as a bound on semi-infinite and infinite segments.\n" \
"\n" \
"This class uses comparison-by-identity rather than\n" \
"comparison-by-value.  What this means, is there are only ever two\n" \
"instances of this class, representing positive and negative\n" \
"infinity respectively.  All other \"instances\" of this class are\n" \
"infact simply references to one of these two, and comparisons are\n" \
"done by checking which one you've got.  This improves speed and\n" \
"reduces memory use, and is similar in implementation to Python's\n" \
"boolean True and False objects.\n" \
"\n" \
"The normal way to obtain references to positive or negative\n" \
"infinity is to do infinity() or -infinity() respectively.  It is\n" \
"also possible to select the sign by passing a single numeric\n" \
"argument to the constructor.  The sign of the argument causes a\n" \
"reference to either positive or negative infinity to be returned,\n" \
"respectively.  For example infinity(-1) is equivalent to\n" \
"-infinity().  However, this feature is a little slower and not\n" \
"recommended for normal use;  it is provided only to simplify the\n" \
"pickling and unpickling of instances of the class.\n" \
"\n" \
"Example:\n" \
"\n" \
">>> x = infinity()\n" \
">>> x > 0\n" \
"True\n" \
">>> x + 10 == x\n" \
"True\n" \
">>> segment(-10, 10) - segment(-x, 0)\n" \
"segment(0, 10)",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES,
	.tp_methods = methods,
	.tp_name = MODULE_NAME ".infinity",
	.tp_new = __new__,
	.tp_repr = __repr__,
	.tp_richcompare = richcompare,
	.tp_str = __repr__,
	.tp_hash = (hashfunc) _Py_HashPointer,
};
