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
 *                Segments Module Component --- segment Class
 *
 * ============================================================================
 */


#include <Python.h>
#include <stdlib.h>


#include <segments.h>


/*
 * ============================================================================
 *
 *                               segment Class
 *
 * ============================================================================
 */


/*
 * Utilities
 */


static int segments_Segment_Check(PyObject *obj)
{
	return obj ? PyObject_TypeCheck(obj, &segments_Segment_Type) : 0;
}


/*
 * Basic methods
 */


PyObject *segments_Segment_New(PyTypeObject *type, PyObject *a, PyObject *b)
{
	PyObject *new = type->tp_alloc(type, 2);
	int delta;
	if(new && PyObject_Cmp(a, b, &delta) >= 0) {
		if(delta <= 0) {
			PyTuple_SET_ITEM(new, 0, a);
			PyTuple_SET_ITEM(new, 1, b);
		} else {
			PyTuple_SET_ITEM(new, 0, b);
			PyTuple_SET_ITEM(new, 1, a);
		}
	} else {
		Py_XDECREF(new);
		new = NULL;
		Py_DECREF(a);
		Py_DECREF(b);
	}
	return new;
}


static PyObject *__new__(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
	PyObject *a, *b;

	if(!PyArg_ParseTuple(args, "OO", &a, &b)) {
		PyErr_Clear();
		if(!PyArg_ParseTuple(args, "(OO)", &a, &b)) {
			PyErr_SetString(PyExc_TypeError, "__new__() takes 2 arguments, or 1 arguments when it is a sequence of length 2");
			return NULL;
		}
	}

	Py_INCREF(a);
	Py_INCREF(b);

	return segments_Segment_New(type, a, b);
}


static PyObject *__repr__(PyObject *self)
{
	PyObject *a = PyObject_Repr(PyTuple_GET_ITEM(self, 0));
	PyObject *b = PyObject_Repr(PyTuple_GET_ITEM(self, 1));
	PyObject *result;
	if(a && b)
		result = PyString_FromFormat("Segment(%s, %s)", PyString_AsString(a), PyString_AsString(b));
	else
		result = NULL;
	Py_XDECREF(a);
	Py_XDECREF(b);
	return result;
}


static PyObject *__str__(PyObject *self)
{
	PyObject *a = PyObject_Str(PyTuple_GET_ITEM(self, 0));
	PyObject *b = PyObject_Str(PyTuple_GET_ITEM(self, 1));
	PyObject *result;
	if(a && b)
		result = PyString_FromFormat("[%s ... %s)", PyString_AsString(a), PyString_AsString(b));
	else
		result = NULL;
	Py_XDECREF(a);
	Py_XDECREF(b);
	return result;
}


/*
 * Accessors
 */


static PyObject *__abs__(PyObject *self)
{
	return PyNumber_Subtract(PyTuple_GET_ITEM(self, 1), PyTuple_GET_ITEM(self, 0));
}


/*
 * Comparisons
 */


static int __nonzero__(PyObject *self)
{
	return PyObject_Compare(PyTuple_GET_ITEM(self, 0), PyTuple_GET_ITEM(self, 1)) != 0;
}


static PyObject *richcompare(PyObject *self, PyObject *other, int op_id)
{
	if(!PyTuple_Check(other)) {
		PyObject *sa = PyTuple_GET_ITEM(self, 0);
		PyObject *result;
		Py_INCREF(sa);
		result = PyObject_RichCompare(sa, other, op_id);
		Py_DECREF(sa);
		return result;
	}
	return PyTuple_Type.tp_richcompare(self, other, op_id);
}


static PyObject *intersects(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	PyObject *oa, *ob;
	PyObject *result;
	if(!segments_Segment_Check(other)) {
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}
	oa = PyTuple_GET_ITEM(other, 0);
	ob = PyTuple_GET_ITEM(other, 1);
	result = (PyObject_Compare(sb, oa) > 0) && (PyObject_Compare(sa, ob) < 0) ? Py_True : Py_False;
	Py_INCREF(result);
	return result;
}


static int __contains__(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	if(segments_Segment_Check(other)) {
		PyObject *oa = PyTuple_GET_ITEM(other, 0);
		PyObject *ob = PyTuple_GET_ITEM(other, 1);
		return (PyObject_Compare(sa, oa) <= 0) && (PyObject_Compare(sb, ob) >= 0);
	} else
		return (PyObject_Compare(sa, other) <= 0) && (PyObject_Compare(other, sb) < 0);
}


static PyObject *disjoint(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	PyObject *oa, *ob;
	if(!segments_Segment_Check(other)) {
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}
	oa = PyTuple_GET_ITEM(other, 0);
	ob = PyTuple_GET_ITEM(other, 1);
	if(PyObject_Compare(sa, ob) > 0)
		return PyInt_FromLong(1);
	if(PyObject_Compare(sb, oa) < 0)
		return PyInt_FromLong(-1);
	return PyInt_FromLong(0);
}


/*
 * Arithmetic
 */


static PyObject *__and__(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	PyObject *oa, *ob;
	PyObject *a, *b;
	if(!segments_Segment_Check(other)) {
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}
	oa = PyTuple_GET_ITEM(other, 0);
	ob = PyTuple_GET_ITEM(other, 1);
	if((PyObject_Compare(sb, oa) <= 0) || (PyObject_Compare(sa, ob) >= 0)) {
		/* self and other don't intersect */
		PyErr_SetObject(PyExc_ValueError, other);
		return NULL;
	}
	a = (PyObject_Compare(sa, oa) >= 0) ? sa : oa;
	b = (PyObject_Compare(sb, ob) <= 0) ? sb : ob;
	if((a == sa) && (b == sb)) {
		/* re-use self */
		Py_INCREF(self);
		return self;
	}
	if((a == oa) && (b == ob)) {
		/* re-use other */
		Py_INCREF(other);
		return other;
	}
	Py_INCREF(a);
	Py_INCREF(b);
	return segments_Segment_New(self->ob_type, a, b);
}


static PyObject *__or__(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	PyObject *oa, *ob;
	PyObject *a, *b;
	if(!segments_Segment_Check(other)) {
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}
	oa = PyTuple_GET_ITEM(other, 0);
	ob = PyTuple_GET_ITEM(other, 1);
	if((PyObject_Compare(sb, oa) < 0) || (PyObject_Compare(sa, ob) > 0)) {
		/* self and other are disjoint */
		PyErr_SetObject(PyExc_ValueError, other);
		return NULL;
	}
	a = (PyObject_Compare(sa, oa) <= 0) ? sa : oa;
	b = (PyObject_Compare(sb, ob) >= 0) ? sb : ob;
	if((a == sa) && (b == sb)) {
		/* re-use self */
		Py_INCREF(self);
		return self;
	}
	if((a == oa) && (b == ob)) {
		/* re-use other */
		Py_INCREF(other);
		return other;
	}
	Py_INCREF(a);
	Py_INCREF(b);
	return segments_Segment_New(self->ob_type, a, b);
}


static PyObject *__sub__(PyObject *self, PyObject *other)
{
	PyObject *sa = PyTuple_GET_ITEM(self, 0);
	PyObject *sb = PyTuple_GET_ITEM(self, 1);
	PyObject *oa, *ob;
	PyObject *a, *b;
	if(!segments_Segment_Check(other)) {
		PyErr_SetObject(PyExc_TypeError, other);
		return NULL;
	}
	oa = PyTuple_GET_ITEM(other, 0);
	ob = PyTuple_GET_ITEM(other, 1);
	if((PyObject_Compare(sb, oa) <= 0) || (PyObject_Compare(sa, ob) >= 0)) {
		/* self and other do not intersect */
		Py_INCREF(self);
		return self;
	}
	if(__contains__(other, self) || ((PyObject_Compare(sa, oa) < 0) && (PyObject_Compare(sb, ob) > 0))) {
		/* result is not exactly 1 segment */
		PyErr_SetObject(PyExc_ValueError, other);
		return NULL;
	}
	if(PyObject_Compare(sa, oa) < 0) {
		a = sa;
		b = oa;
	} else {
		a = ob;
		b = sb;
	}
	Py_INCREF(a);
	Py_INCREF(b);
	return segments_Segment_New(self->ob_type, a, b);
}


/*
 * Protraction and contraction and shifting
 */


static PyObject *protract(PyObject *self, PyObject *delta)
{
	PyObject *a = PyNumber_Subtract(PyTuple_GET_ITEM(self, 0), delta);
	PyObject *b = PyNumber_Add(PyTuple_GET_ITEM(self, 1), delta);
	if(PyErr_Occurred()) {
		Py_XDECREF(a);
		Py_XDECREF(b);
		return NULL;
	}
	return segments_Segment_New(self->ob_type, a, b);
}


static PyObject *contract(PyObject *self, PyObject *delta)
{
	PyObject *a = PyNumber_Add(PyTuple_GET_ITEM(self, 0), delta);
	PyObject *b = PyNumber_Subtract(PyTuple_GET_ITEM(self, 1), delta);
	if(PyErr_Occurred()) {
		Py_XDECREF(a);
		Py_XDECREF(b);
		return NULL;
	}
	return segments_Segment_New(self->ob_type, a, b);
}


static PyObject *shift(PyObject *self, PyObject *delta)
{
	PyObject *a = PyNumber_Add(PyTuple_GET_ITEM(self, 0), delta);
	PyObject *b = PyNumber_Add(PyTuple_GET_ITEM(self, 1), delta);
	if(PyErr_Occurred()) {
		Py_XDECREF(a);
		Py_XDECREF(b);
		return NULL;
	}
	return segments_Segment_New(self->ob_type, a, b);
}


/*
 * Type information
 */


static PyNumberMethods as_number = {
	.nb_add = __or__,
	.nb_and = __and__,
	.nb_absolute = __abs__,
	.nb_nonzero = __nonzero__,
	.nb_or = __or__,
	.nb_subtract = __sub__,
};


static PySequenceMethods as_sequence = {
	.sq_contains = __contains__,
};


static struct PyMethodDef methods[] = {
	{"disjoint", disjoint, METH_O, "Returns >0 if self covers an interval above other's interval, <0 if self covers an interval below other's, or 0 if the two intervals are not disjoint (intersect or touch).  A return value of 0 indicates the two segments would coalesce."},
	{"intersects", intersects, METH_O, "Return True if the intersection of self and other is not a null segment."},
	{"protract", protract, METH_O, "Return a new segment whose bounds are given by subtracting x from the segment's lower bound and adding x to the segment's upper bound."},
	{"contract", contract, METH_O, "Return a new segment whose bounds are given by adding x to the segment's lower bound and subtracting x from the segment's upper bound."},
	{"shift", shift, METH_O, "Return a new segment whose bounds are given by adding x to the segment's upper and lower bounds."},
	{NULL,}
};


PyTypeObject segments_Segment_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_base = &PyTuple_Type,
	.tp_as_number = &as_number,
	.tp_as_sequence = &as_sequence,
	.tp_doc =
"The Segment class defines objects that represent a range of values.\n" \
"A segment has a start and an end, and is taken to represent the\n" \
"range of values in the semi-open interval [start, end).  Some\n" \
"limited arithmetic operations are possible with Segments, but\n" \
"because the set of (single) Segments is not closed under the\n" \
"sensible definitions of the standard arithmetic operations, the\n" \
"behaviour of the arithmetic operators on Segments may not be as you\n" \
"would expect.  For general arithmetic on Segments, use SegmentList\n" \
"objects.  The methods for this class exist mostly for purpose of\n" \
"simplifying the implementation of the SegmentList class.\n" \
"\n" \
"The Segment class is a subclass of the tuple built-in class\n" \
"provided by Python.  This means Segments are immutable --- you\n" \
"cannot modify a Segment object after creating it, to change the\n" \
"boundaries of a Segment you must create a new Segment object with\n" \
"the desired boundaries.  Like tuples, Segments can be used as\n" \
"dictionary keys, and like tuples the comparison used to find a\n" \
"Segment in the dictionary is done by value not by ID.  And, like\n" \
"tuples, a Segment can be created from any sequence-like object by\n" \
"passing it to the constructor (the sequence must have exactly two\n" \
"elements in it).\n" \
"\n" \
"Example::\n" \
"\n" \
"    >>> Segment(0, 10) & Segment(5, 15)\n" \
"    segment(5, 10)\n" \
"    >>> segment(0, 10) | segment(5, 15)\n" \
"    segment(0, 15)\n" \
"    >>> segment(0, 10) - segment(5, 15)\n" \
"    segment(0, 5)\n" \
"    >>> segment(0, 10) < segment(5, 15)\n" \
"    True\n" \
"    >>> segment(1, 2) in segment(0, 10)\n" \
"    True\n" \
"    >>> bool(segment(0, 0))\n" \
"    False\n" \
"    >>> segment(\"AAA Towing\", \"York University\") & segment(\"Pool\", \"Zoo\")\n" \
"    segment('Pool', 'York University')\n" \
"    >>> x = [0, 1]\n" \
"    >>> segment(x)\n" \
"    segment(0, 1)\n" \
"    >>> y = segment(0, 1)\n" \
"    >>> y == x\n" \
"    True\n" \
"    >>> y is x\n" \
"    False\n" \
"    >>> z = {x: [\"/path/to/file1\", \"/path/to/file2\"]}\n" \
"    >>> y in z\n" \
"    True\n" \
"    >>> z[y]\n" \
"    ['/path/to/file1', '/path/to/file2']",
	.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_CHECKTYPES | Py_TPFLAGS_BASETYPE,
	.tp_methods = methods,
	.tp_name = MODULE_NAME ".Segment",
	.tp_new = __new__,
	.tp_repr = __repr__,
	.tp_str = __str__,
	.tp_richcompare = richcompare,
	/* The compiler doesn't like the following, so instead this is done
	 * at runtime in the module init() function (in segments.c) just
	 * before creating the type */
	/* .tp_hash = PyTuple_Type.tp_hash, */
};
