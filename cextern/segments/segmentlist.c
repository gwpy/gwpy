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
 *              Segments Module Component --- segmentlist Class
 *
 * ============================================================================
 */


#include <Python.h>
#include <stdlib.h>


#include <segments.h>

/* Gain access to 64-bit addressing where possible
 * http://www.python.org/dev/peps/pep-0353/#conversion-guidelines */
#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

/*
 * ============================================================================
 *
 *                             segmentlist Class
 *
 * ============================================================================
 */


/*
 * Utilities
 */


/* commented out to silence compiler warnings, but it might be needed so I
 * don't want to just delete it */
#if 0
static int segments_SegmentList_Check(PyObject *obj)
{
	return obj ? PyObject_TypeCheck(obj, &segments_SegmentList_Type) : 0;
}
#endif


/* copied from bisect.py */

static Py_ssize_t bisect_left(PyObject *seglist, PyObject *seg, Py_ssize_t lo, Py_ssize_t hi)
{
	if(lo < 0)
		lo = 0;

	if(hi < 0) {
		hi = PyList_GET_SIZE(seglist);
		if(hi < 0)
			return -1;
	}

	while(lo < hi) {
		Py_ssize_t mid = (lo + hi) / 2;
		PyObject *item = PyList_GET_ITEM(seglist, mid);
		Py_ssize_t result;
		if(!item)
			return -1;
		Py_INCREF(item);
		result = PyObject_RichCompareBool(item, seg, Py_LT);
		Py_DECREF(item);
		if(result < 0)
			/* error */
			return -1;
		else if(result > 0)
			/* item < seg */
			lo = mid + 1;
		else
			/* item >= seg */
			hi = mid;
	}

	return lo;
}


/* copied from bisect.py */

static Py_ssize_t bisect_right(PyObject *seglist, PyObject *seg, Py_ssize_t lo, Py_ssize_t hi)
{
	if(lo < 0)
		lo = 0;

	if(hi < 0) {
		hi = PyList_GET_SIZE(seglist);
		if(hi < 0)
			return -1;
	}

	while(lo < hi) {
		Py_ssize_t mid = (lo + hi) / 2;
		PyObject *item = PyList_GET_ITEM(seglist, mid);
		Py_ssize_t result;
		if(!item)
			return -1;
		Py_INCREF(item);
		result = PyObject_RichCompareBool(seg, item, Py_LT);
		Py_DECREF(item);
		if(result < 0)
			/* error */
			return -1;
		else if(result > 0)
			/* seg < item */
			hi = mid;
		else
			/* seg >= item */
			lo = mid + 1;
	}

	return lo;
}


static int unpack(PyObject *seg, PyObject **lo, PyObject **hi)
{
	if(!seg)
		return -1;
	if(!PyTuple_Check(seg)) {
		PyErr_SetObject(PyExc_TypeError, seg);
		return -1;
	}

	if(lo) {
		*lo = PyTuple_GetItem(seg, 0);
		if(!*lo) {
			if(hi)
				*hi = NULL;
			return -1;
		}
		Py_INCREF(*lo);
	}

	if(hi) {
		*hi = PyTuple_GetItem(seg, 1);
		if(!*hi) {
			if(lo) {
				Py_XDECREF(*lo);
				*lo = NULL;
			}
			return -1;
		}
		Py_INCREF(*hi);
	}

	return 0;
}


static PyObject *min(PyObject *a, PyObject *b)
{
	int result;

	result = PyObject_RichCompareBool(a, b, Py_LT);
	if(result < 0) {
		Py_DECREF(a);
		Py_DECREF(b);
		return NULL;
	} else if(result > 0) {
		Py_DECREF(b);
		return a;
	} else {
		Py_DECREF(a);
		return b;
	}
}


static PyObject *max(PyObject *a, PyObject *b)
{
	int result;

	result = PyObject_RichCompareBool(a, b, Py_GT);
	if(result < 0) {
		Py_DECREF(a);
		Py_DECREF(b);
		return NULL;
	} else if(result > 0) {
		Py_DECREF(b);
		return a;
	} else {
		Py_DECREF(a);
		return b;
	}
}


static PyObject *make_segment(PyObject *lo, PyObject *hi)
{
	return segments_Segment_New(&segments_Segment_Type, lo, hi);
}


static int pylist_extend(PyListObject *l, PyObject *v)
{
#if (PY_MAJOR_VERSION >= 2) && (PY_MINOR_VERSION >= 4)
	PyObject *result = _PyList_Extend(l, v);
#else
	PyObject *result = PyObject_CallMethod((PyObject *) l, "extend", "O", v);
#endif
	if(!result)
		return -1;
	Py_DECREF(result);
	return 0;
}


static PyListObject *segments_SegmentList_New(PyTypeObject *type, PyObject *sequence)
{
	PyListObject *new = (PyListObject *) type->tp_alloc(type, 0);
	if(new && sequence)
		if(pylist_extend(new, sequence)) {
			Py_DECREF(new);
			new = NULL;
		}
	return new;
}


/*
 * Accessors
 */


static PyObject *__abs__(PyObject *self)
{
	Py_ssize_t i;
	PyObject *abs;

	abs = PyInt_FromLong(0);
	if(!abs)
		return NULL;

	for(i = 0; i < PyList_GET_SIZE(self); i++) {
		PyObject *seg, *segsize, *newabs;
		seg = PyList_GET_ITEM(self, i);
		if(!seg) {
			Py_DECREF(abs);
			return NULL;
		}
		Py_INCREF(seg);
		segsize = PyNumber_Absolute(seg);
		Py_DECREF(seg);
		if(!segsize) {
			Py_DECREF(abs);
			return NULL;
		}
		newabs = PyNumber_InPlaceAdd(abs, segsize);
		Py_DECREF(segsize);
		Py_DECREF(abs);
		abs = newabs;
		if(!abs)
			return NULL;
	}

	return abs;
}


static PyObject *extent(PyObject *self, PyObject *nul)
{
	Py_ssize_t n = PyList_GET_SIZE(self);
	Py_ssize_t i;
	PyObject *lo, *hi;

	if(n < 0)
		return NULL;
	if(n < 1) {
		PyErr_SetString(PyExc_ValueError, "empty list");
		return NULL;
	}

	if(unpack(PyList_GET_ITEM(self, 0), &lo, &hi))
		return NULL;

	for(i = 1; i < n; i++) {
		PyObject *item_lo, *item_hi;

		if(unpack(PyList_GET_ITEM(self, i), &item_lo, &item_hi)) {
			Py_DECREF(lo);
			Py_DECREF(hi);
			return NULL;
		}

		lo = min(lo, item_lo);
		if(!lo) {
			Py_DECREF(hi);
			Py_DECREF(item_hi);
			return NULL;
		}

		hi = max(hi, item_hi);
		if(!hi) {
			Py_DECREF(lo);
			Py_DECREF(item_lo);
			return NULL;
		}
	}

	return make_segment(lo, hi);
}


static PyObject *find(PyObject *self, PyObject *item)
{
	Py_ssize_t n = PyList_GET_SIZE(self);
	Py_ssize_t i;

	if(n < 0)
		return NULL;

	Py_INCREF(item);
	for(i = 0; i < n; i++) {
		Py_ssize_t result;
		PyObject *seg = PyList_GET_ITEM(self, i);
		Py_INCREF(seg);
		result = PySequence_Contains(seg, item);
		Py_DECREF(seg);
		if(result < 0) {
			Py_DECREF(item);
			return NULL;
		} else if(result > 0) {
			Py_DECREF(item);
			/* match found */
			return PyInt_FromLong(i);
		}
	}
	Py_DECREF(item);

	PyErr_SetObject(PyExc_ValueError, item);
	return NULL;
}


/*
 * Comparisons
 */


static PyObject *intersects(PyObject *self, PyObject *other)
{
	Py_ssize_t n_self = PyList_GET_SIZE(self);
	Py_ssize_t n_other = PySequence_Size(other);
	PyObject *seg;
	PyObject *lo;
	PyObject *hi;
	PyObject *olo;
	PyObject *ohi;
	int result;
	Py_ssize_t i, j;

	if((n_self < 0) || (n_other < 0))
		return NULL;
	if((n_self < 1) || (n_other < 1)) {
		Py_INCREF(Py_False);
		return Py_False;
	}

	i = j = 0;

	if(unpack(PyList_GET_ITEM(self, 0), &lo, &hi))
		return NULL;
	seg = PySequence_GetItem(other, 0);
	if(unpack(seg, &olo, &ohi)) {
		Py_DECREF(lo);
		Py_DECREF(hi);
		Py_XDECREF(seg);
		return NULL;
	}
	Py_DECREF(seg);

	while(1) {
		if((result = PyObject_RichCompareBool(hi, olo, Py_LE)) < 0) {
			Py_DECREF(lo);
			Py_DECREF(hi);
			Py_DECREF(olo);
			Py_DECREF(ohi);
			return NULL;
		} else if(result > 0) {
			Py_DECREF(lo);
			Py_DECREF(hi);
			if(++i >= n_self) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				Py_INCREF(Py_False);
				return Py_False;
			}
			if(unpack(PyList_GET_ITEM(self, i), &lo, &hi)) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				return NULL;
			}
		} else if((result = PyObject_RichCompareBool(ohi, lo, Py_LE)) < 0) {
			Py_DECREF(lo);
			Py_DECREF(hi);
			Py_DECREF(olo);
			Py_DECREF(ohi);
			return NULL;
		} else if(result > 0) {
			Py_DECREF(olo);
			Py_DECREF(ohi);
			if(++j >= n_other) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_INCREF(Py_False);
				return Py_False;
			}
			seg = PySequence_GetItem(other, j);
			if(unpack(seg, &olo, &ohi)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_XDECREF(seg);
				return NULL;
			}
			Py_DECREF(seg);
		} else {
			/* self[i] and other[j] intersect */
			Py_DECREF(lo);
			Py_DECREF(hi);
			Py_DECREF(olo);
			Py_DECREF(ohi);
			Py_INCREF(Py_True);
			return Py_True;
		}
	}
}


static PyObject *intersects_segment(PyObject *self, PyObject *other)
{
	Py_ssize_t i = bisect_left(self, other, -1, -1);
	PyObject *a = NULL, *b = NULL;
	int result;

	if(i < 0)
		/* error */
		return NULL;

	if(i != 0) {
		if(unpack(other, &a, NULL) || unpack(PyList_GET_ITEM(self, i - 1), NULL, &b)) {
			Py_XDECREF(a);
			Py_XDECREF(b);
			return NULL;
		}
		result = PyObject_RichCompareBool(a, b, Py_LT);
		Py_DECREF(a);
		Py_DECREF(b);
		if(result < 0)
			return NULL;
		else if(result > 0) {
			Py_INCREF(Py_True);
			return Py_True;
		}
	}

	if(i != PyList_GET_SIZE(self)) {
		if(unpack(other, NULL, &a) || unpack(PyList_GET_ITEM(self, i), &b, NULL)) {
			Py_XDECREF(a);
			Py_XDECREF(b);
			return NULL;
		}
		result = PyObject_RichCompareBool(a, b, Py_GT);
		Py_DECREF(a);
		Py_DECREF(b);
		if(result < 0)
			return NULL;
		else if(result > 0) {
			Py_INCREF(Py_True);
			return Py_True;
		}
	}

	Py_INCREF(Py_False);
	return Py_False;
}


static int __contains__(PyObject *self, PyObject *other)
{
	Py_ssize_t i;
	Py_ssize_t result;

	if(PyObject_TypeCheck(other, self->ob_type)) {
		for(i = 0; i < PyList_GET_SIZE(other); i++) {
			PyObject *seg = PyList_GET_ITEM(other, i);
			Py_INCREF(seg);
			result = __contains__(self, seg);
			Py_DECREF(seg);
			if(result <= 0)
				return result;
		}
		return 1;
	}

	i = bisect_left(self, other, -1, -1);
	if(i < 0)
		/* error */
		return i;

	if(i != 0) {
		PyObject *seg = PyList_GET_ITEM(self, i - 1);
		if(!seg)
			return -1;
		Py_INCREF(seg);
		result = PySequence_Contains(seg, other);
		Py_DECREF(seg);
		if(result)
			return result > 0 ? 1 : result;
	}

	if(i != PyList_GET_SIZE(self)) {
		PyObject *seg = PyList_GET_ITEM(self, i);
		if(!seg)
			return -1;
		Py_INCREF(seg);
		result = PySequence_Contains(seg, other);
		Py_DECREF(seg);
		if(result)
			return result > 0 ? 1 : result;
	}

	return 0;
}


/*
 * Coalesce
 */


static PyObject *coalesce(PyObject *self, PyObject *nul)
{
	PyObject *lo, *hi;
	int result;
	Py_ssize_t i, j;
	Py_ssize_t n;

	if(PyList_Sort(self) < 0)
		return NULL;

	n = PyList_GET_SIZE(self);
	if(n < 0)
		return NULL;

	i = j = 0;
	while(j < n) {
		if(unpack(PyList_GET_ITEM(self, j++), &lo, &hi))
			return NULL;

		result = 0;
		while(j < n) {
			PyObject *a, *b;
			if(unpack(PyList_GET_ITEM(self, j), &a, &b)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				return NULL;
			}
			result = PyObject_RichCompareBool(hi, a, Py_GE);
			Py_DECREF(a);
			if(result < 0) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(b);
				return NULL;
			} else if(result > 0) {
				hi = max(hi, b);
				if(!hi) {
					Py_DECREF(lo);
					return NULL;
				}
				j++;
			} else {
				Py_DECREF(b);
				break;
			}
		}

		if((result = PyObject_RichCompareBool(lo, hi, Py_NE)) < 0) {
			Py_DECREF(lo);
			Py_DECREF(hi);
			return NULL;
		} else if(result > 0) {
			PyObject *seg = make_segment(lo, hi);
			if(!seg)
				return NULL;
			/* _SetItem consumes a ref count */
			if(PyList_SetItem(self, i, seg) < 0) {
				Py_DECREF(seg);
				return NULL;
			}
			i++;
		} else {
			Py_DECREF(lo);
			Py_DECREF(hi);
		}

	}

	if(PyList_SetSlice(self, i, n, NULL) < 0)
		return NULL;

	Py_INCREF(self);
	return self;
}


/*
 * Arithmetic
 */


static PyObject *__iand__(PyObject *self, PyObject *other)
{
	PyObject *new = NULL;
	other = PyNumber_Invert(other);
	if(other) {
		new = PyNumber_InPlaceSubtract(self, other);
		Py_DECREF(other);
	}
	return new;
}


static PyObject *__and__(PyObject *self, PyObject *other)
{
	PyObject *new = NULL;

	/* error checking on size functions not required */
	if(PyList_GET_SIZE(self) >= PySequence_Size(other)) {
		self = (PyObject *) segments_SegmentList_New(self->ob_type, self);
		if(self) {
			new = PyNumber_InPlaceAnd(self, other);
			Py_DECREF(self);
		}
	} else {
		other = (PyObject *) segments_SegmentList_New(self->ob_type, other);
		if(other) {
			new = PyNumber_InPlaceAnd(other, self);
			Py_DECREF(other);
		}
	}

	return new;
}


static PyObject *__ior__(PyObject *self, PyObject *other)
{
	PyObject *seg, *lo, *hi;
	int result;
	Py_ssize_t i, j;
	Py_ssize_t n;

	/* Faster algorithm when the two lists have very different sizes.
	 * OK to not test size functions for error return values */
	if(PySequence_Size(other) > PyList_GET_SIZE(self) / 2) {
		if(pylist_extend((PyListObject *) self, other))
			return NULL;
		return coalesce(self, NULL);
	}

	/* don't iterate over the same object twice */
	if(other == self) {
		Py_INCREF(self);
		return self;
	}

	i = 0;
	other = PyObject_GetIter(other);
	if(!other)
		return NULL;
	while((seg = PyIter_Next(other))) {
		PyObject *item_lo, *item_hi;

		i = j = bisect_right(self, seg, i, -1);
		if(i < 0) {
			Py_DECREF(seg);
			Py_DECREF(other);
			return NULL;
		}
		if(unpack(seg, &lo, &hi)) {
			Py_DECREF(seg);
			Py_DECREF(other);
			return NULL;
		}

		if(i > 0) {
			if(unpack(PyList_GET_ITEM(self, i - 1), &item_lo, &item_hi)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			}
			if((result = PyObject_RichCompareBool(item_hi, lo, Py_GE)) < 0) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(item_lo);
				Py_DECREF(item_hi);
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			} else if(result > 0) {
				i--;
				Py_DECREF(lo);
				lo = item_lo;
			} else {
				Py_DECREF(item_lo);
			}
			Py_DECREF(item_hi);
		}

		n = PyList_GET_SIZE(self);
		while(j < n) {
			if(unpack(PyList_GET_ITEM(self, j), &item_lo, NULL)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			} else if((result = PyObject_RichCompareBool(item_lo, hi, Py_LE)) < 0) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(item_lo);
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			}
			Py_DECREF(item_lo);
			if(result > 0) {
				j++;
			} else {
				break;
			}
		}

		if(j > i) {
			Py_DECREF(seg);
			if(unpack(PyList_GET_ITEM(self, j - 1), NULL, &item_hi)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_DECREF(other);
				return NULL;
			}
			hi = max(hi, item_hi);
			if(!hi) {
				Py_DECREF(lo);
				Py_DECREF(other);
				return NULL;
			}
			seg = make_segment(lo, hi);
			if(!seg) {
				Py_DECREF(other);
				return NULL;
			}
			if(PyList_SetSlice(self, i + 1, j, NULL) < 0) {
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			}
			/* _SetItem consumes a ref count */
			if(PyList_SetItem(self, i, seg) < 0) {
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			}
		} else {
			Py_DECREF(lo);
			Py_DECREF(hi);
			/* _Insert increments seg's ref count */
			if(PyList_Insert(self, i, seg) < 0) {
				Py_DECREF(seg);
				Py_DECREF(other);
				return NULL;
			}
			Py_DECREF(seg);
		}
		i++;
	}
	Py_DECREF(other);
	if(PyErr_Occurred())
		return NULL;

	Py_INCREF(self);
	return self;
}


static PyObject *__or__(PyObject *self, PyObject *other)
{
	PyObject *new = NULL;

	/* error checking on size functions not required */
	if(PyList_GET_SIZE(self) >= PySequence_Size(other)) {
		self = (PyObject *) segments_SegmentList_New(self->ob_type, self);
		if(self) {
			new = PyNumber_InPlaceOr(self, other);
			Py_DECREF(self);
		}
	} else {
		other = (PyObject *) segments_SegmentList_New(self->ob_type, other);
		if(other) {
			new = PyNumber_InPlaceOr(other, self);
			Py_DECREF(other);
		}
	}
	return new;
}


static PyObject *__xor__(PyObject *self, PyObject *other)
{
	PyObject *new;

	new = PyNumber_Subtract(self, other);
	other = PyNumber_Subtract(other, self);
	if(!(new && other)) {
		Py_XDECREF(new);
		Py_XDECREF(other);
		return NULL;
	}
	if(pylist_extend((PyListObject *) new, other)) {
		Py_DECREF(new);
		Py_DECREF(other);
		return NULL;
	}
	Py_DECREF(other);

	if(PyList_Sort(new) < 0) {
		Py_DECREF(new);
		return NULL;
	}

	return new;
}


static PyObject *__isub__(PyObject *self, PyObject *other)
{
	PyObject *seg;
	PyObject *olo, *ohi;
	PyObject *lo, *hi;
	int result;
	Py_ssize_t i, j;
	Py_ssize_t n;
	
	n = PySequence_Size(other);
	if(n < 0)
		return NULL;
	if(n < 1) {
		Py_INCREF(self);
		return self;
	}

	/* don't iterate over the same object twice */
	if(other == self) {
		PySequence_DelSlice(self, 0, n);
		Py_INCREF(self);
		return self;
	}

	i = j = 0;

	seg = PySequence_GetItem(other, j);
	if(unpack(seg, &olo, &ohi)) {
		Py_XDECREF(seg);
		return NULL;
	}
	Py_DECREF(seg);

	while(i < PyList_GET_SIZE(self)) {
		if(unpack(PyList_GET_ITEM(self, i), &lo, &hi)) {
			Py_DECREF(olo);
			Py_DECREF(ohi);
			return NULL;
		}

		while((result = PyObject_RichCompareBool(ohi, lo, Py_LE))) {
			if(result < 0) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				Py_DECREF(lo);
				Py_DECREF(hi);
				return NULL;
			}
			if(++j >= n) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_INCREF(self);
				return self;
			}
			Py_DECREF(olo);
			Py_DECREF(ohi);
			seg = PySequence_GetItem(other, j);
			if(unpack(seg, &olo, &ohi)) {
				Py_DECREF(lo);
				Py_DECREF(hi);
				Py_XDECREF(seg);
				return NULL;
			}
			Py_DECREF(seg);
		}

		if((result = PyObject_RichCompareBool(hi, olo, Py_LE)) < 0) {
			Py_DECREF(olo);
			Py_DECREF(ohi);
			Py_DECREF(lo);
			Py_DECREF(hi);
			return NULL;
		} else if(result > 0) {
			/* seg[1] <= otherseg[0] */
			i++;
		} else if((result = PyObject_RichCompareBool(olo, lo, Py_LE)) < 0) {
			Py_DECREF(olo);
			Py_DECREF(ohi);
			Py_DECREF(lo);
			Py_DECREF(hi);
			return NULL;
		} else if(result > 0) {
			/* otherseg[0] <= seg[0] */
			if((result = PyObject_RichCompareBool(ohi, hi, Py_GE)) < 0) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				Py_DECREF(lo);
				Py_DECREF(hi);
				return NULL;
			} else if(result > 0) {
				/* otherseg[1] >= seg[1] */
				if(PySequence_DelItem(self, i) < 0) {
					Py_DECREF(olo);
					Py_DECREF(ohi);
					Py_DECREF(lo);
					Py_DECREF(hi);
					return NULL;
				}
			} else {
				/* else */
				PyObject *newseg = make_segment(ohi, hi);
				if(!newseg) {
					Py_DECREF(olo);
					Py_DECREF(lo);
					return NULL;
				}
				/* _SetItem consumes a ref count */
				if(PyList_SetItem(self, i, newseg) < 0) {
					Py_DECREF(olo);
					Py_DECREF(lo);
					Py_DECREF(newseg);
					return NULL;
				}
				/* make_segment() consumed references,
				 * which we need */
				Py_INCREF(ohi);
				Py_INCREF(hi);
			}
		} else {
			/* else */
			PyObject *newseg = make_segment(lo, olo);
			if(!newseg) {
				Py_DECREF(ohi);
				Py_DECREF(hi);
				return NULL;
			}
			/* _SetItem consumes a ref count */
			if(PyList_SetItem(self, i++, newseg) < 0) {
				Py_DECREF(ohi);
				Py_DECREF(hi);
				Py_DECREF(newseg);
				return NULL;
			}
			/* make_segment() consumed references, which we
			 * need */
			Py_INCREF(lo);
			Py_INCREF(olo);
			if((result = PyObject_RichCompareBool(ohi, hi, Py_LT)) < 0) {
				Py_DECREF(olo);
				Py_DECREF(ohi);
				Py_DECREF(lo);
				Py_DECREF(hi);
				return NULL;
			} else if(result > 0) {
				/* otherseg[1] < seg[1] */
				newseg = make_segment(ohi, hi);
				if(!newseg) {
					Py_DECREF(olo);
					Py_DECREF(lo);
					return NULL;
				}
				/* _Insert increments the ref count */
				if(PyList_Insert(self, i, newseg) < 0) {
					Py_DECREF(olo);
					Py_DECREF(lo);
					Py_DECREF(newseg);
					return NULL;
				}
				Py_DECREF(newseg);
				/* make_segment() consumed references,
				 * which we need */
				Py_INCREF(ohi);
				Py_INCREF(hi);
			}
		}
		Py_DECREF(lo);
		Py_DECREF(hi);
	}
	Py_DECREF(olo);
	Py_DECREF(ohi);

	Py_INCREF(self);
	return self;
}


static PyObject *__sub__(PyObject *self, PyObject *other)
{
	PyObject *new = NULL;
	self = (PyObject *) segments_SegmentList_New(self->ob_type, self);
	if(self) {
		new = PyNumber_InPlaceSubtract(self, other);
		Py_DECREF(self);
	}
	return new;
}


static PyObject *__invert__(PyObject *self)
{
	PyObject *seg, *newseg;
	PyObject *a, *last;
	PyObject *new;
	int result;
	Py_ssize_t i;
	Py_ssize_t n;

	n = PyList_GET_SIZE(self);
	if(n < 0)
		return NULL;

	new = (PyObject *) segments_SegmentList_New(self->ob_type, NULL);
	if(!new)
		return NULL;

	if(n < 1) {
		Py_INCREF(segments_NegInfinity);
		Py_INCREF(segments_PosInfinity);
		newseg = make_segment((PyObject *) segments_NegInfinity, (PyObject *) segments_PosInfinity);
		if(!newseg) {
			Py_DECREF(new);
			return NULL;
		}
		/* _Append increments newseg's ref count */
		if(PyList_Append(new, newseg) < 0) {
			Py_DECREF(newseg);
			Py_DECREF(new);
			return NULL;
		}
		Py_DECREF(newseg);
		return new;
	}

	if(unpack(seg = PyList_GET_ITEM(self, 0), &a, NULL)) {
		Py_DECREF(new);
		return NULL;
	}
	Py_INCREF(segments_NegInfinity);
	if((result = PyObject_RichCompareBool(a, (PyObject *) segments_NegInfinity, Py_GT)) < 0) {
		Py_DECREF(segments_NegInfinity);
		Py_DECREF(a);
		Py_DECREF(new);
		return NULL;
	} else if(result > 0) {
		newseg = make_segment((PyObject *) segments_NegInfinity, a);
		if(!newseg) {
			Py_DECREF(new);
			return NULL;
		}
		/* _Append increments newseg's ref count */
		if(PyList_Append(new, newseg) < 0) {
			Py_DECREF(newseg);
			Py_DECREF(new);
			return NULL;
		}
		Py_DECREF(newseg);
	} else {
		Py_DECREF(segments_NegInfinity);
		Py_DECREF(a);
	}

	if(unpack(seg, NULL, &last)) {
		Py_DECREF(new);
		return NULL;
	}

	for(i = 1; i < n; i++) {
		if(unpack(PyList_GET_ITEM(self, i), &a, NULL)) {
			Py_DECREF(last);
			Py_DECREF(new);
			return NULL;
		}
		newseg = make_segment(last, a);
		if(!newseg) {
			Py_DECREF(new);
			return NULL;
		}
		/* _Append increments newseg's ref count */
		if(PyList_Append(new, newseg) < 0) {
			Py_DECREF(newseg);
			Py_DECREF(new);
			return NULL;
		}
		Py_DECREF(newseg);

		if(unpack(PyList_GET_ITEM(self, i), NULL, &last)) {
			Py_DECREF(new);
			return NULL;
		}
	}

	Py_INCREF(segments_PosInfinity);
	if((result = PyObject_RichCompareBool(last, (PyObject *) segments_PosInfinity, Py_LT)) < 0) {
		Py_DECREF(last);
		Py_DECREF(segments_PosInfinity);
		Py_DECREF(new);
		return NULL;
	} else if(result > 0) {
		newseg = make_segment(last, (PyObject *) segments_PosInfinity);
		if(!newseg) {
			Py_DECREF(new);
			return NULL;
		}
		/* _Append increments newseg's ref count */
		if(PyList_Append(new, newseg) < 0) {
			Py_DECREF(newseg);
			Py_DECREF(new);
			return NULL;
		}
		Py_DECREF(newseg);
	} else {
		Py_DECREF(last);
		Py_DECREF(segments_PosInfinity);
	}

	return new;
}


/*
 * Protraction and contraction and shifting
 */


static PyObject *protract(PyObject *self, PyObject *delta)
{
	PyObject *protract;
	PyObject *seg, *new;
	Py_ssize_t i;
	Py_ssize_t n;

	n = PyList_GET_SIZE(self);
	if(n < 0)
		return NULL;

	protract = PyString_FromString("protract");
	if(!protract)
		return NULL;

	for(i = 0; i < n; i++) {
		seg = PyList_GET_ITEM(self, i);
		if(!seg) {
			Py_DECREF(protract);
			return NULL;
		}
		new = PyObject_CallMethodObjArgs(seg, protract, delta, NULL);
		if(!new) {
			Py_DECREF(protract);
			return NULL;
		}
		/* _SetItem consumes a ref count */
		if(PyList_SetItem(self, i, new) < 0) {
			Py_DECREF(protract);
			return NULL;
		}
	}

	Py_DECREF(protract);

	return coalesce(self, NULL);
}


static PyObject *contract(PyObject *self, PyObject *delta)
{
	PyObject *contract;
	PyObject *seg, *new;
	Py_ssize_t i;
	Py_ssize_t n;

	n = PyList_GET_SIZE(self);
	if(n < 0)
		return NULL;

	contract = PyString_FromString("contract");
	if(!contract)
		return NULL;

	for(i = 0; i < n; i++) {
		seg = PyList_GET_ITEM(self, i);
		if(!seg) {
			Py_DECREF(contract);
			return NULL;
		}
		new = PyObject_CallMethodObjArgs(seg, contract, delta, NULL);
		if(!new) {
			Py_DECREF(contract);
			return NULL;
		}
		/* _SetItem consumes a ref count */
		if(PyList_SetItem(self, i, new) < 0) {
			Py_DECREF(contract);
			return NULL;
		}
	}

	Py_DECREF(contract);

	return coalesce(self, NULL);
}


static PyObject *shift(PyObject *self, PyObject *delta)
{
	PyObject *shift;
	PyObject *seg, *new;
	Py_ssize_t i;
	Py_ssize_t n;

	n = PyList_GET_SIZE(self);
	if(n < 0)
		return NULL;

	shift = PyString_FromString("shift");
	if(!shift)
		return NULL;

	for(i = 0; i < n; i++) {
		seg = PyList_GET_ITEM(self, i);
		if(!seg) {
			Py_DECREF(shift);
			return NULL;
		}
		new = PyObject_CallMethodObjArgs(seg, shift, delta, NULL);
		if(!new) {
			Py_DECREF(shift);
			return NULL;
		}
		/* _SetItem consumes a ref count */
		if(PyList_SetItem(self, i, new) < 0) {
			Py_DECREF(shift);
			return NULL;
		}
	}

	Py_DECREF(shift);

	Py_INCREF(self);
	return self;
}


/*
 * Type information
 */


static PyNumberMethods as_number = {
	.nb_inplace_and = __iand__,
	.nb_and = __and__,
	.nb_inplace_or = __ior__,
	.nb_or = __or__,
	.nb_xor = __xor__,
	.nb_inplace_add = __ior__,
	.nb_add = __or__,
	.nb_inplace_subtract = __isub__,
	.nb_subtract = __sub__,
	.nb_invert = __invert__,
	.nb_absolute = __abs__,
};


static PySequenceMethods as_sequence = {
	.sq_contains = __contains__,
};


static struct PyMethodDef methods[] = {
	{"extent", extent, METH_NOARGS, "Return the segment whose end-points denote the maximum and minimum extent of the segmentlist.  Does not require the segmentlist to be coalesced."},
	{"find", find, METH_O, "Return the smallest i such that i is the index of an element that wholly contains item.  Raises ValueError if no such element exists.  Does not require the segmentlist to be coalesced."},
	{"intersects", intersects, METH_O, "Returns True if the intersection of self and the segmentlist other is not the null set, otherwise returns False.  The algorithm is O(n), but faster than explicit calculation of the intersection, i.e. by testing len(self & other).  Requires both lists to be coalesced."},
	{"intersects_segment", intersects_segment, METH_O, "Returns True if the intersection of self and the segment other is not the null set, otherwise returns False.  The algorithm is O(log n).  Requires the list to be coalesced."},
	{"coalesce", coalesce, METH_NOARGS, "Sort the elements of a list into ascending order, and merge continuous segments into single segments.  This operation is O(n log n)."},
	{"protract", protract, METH_O, "Execute the .protract() method on each segment in the list and coalesce the result.  Segmentlist is modified in place."},
	{"contract", contract, METH_O, "Execute the .contract() method on each segment in the list and coalesce the result.  Segmentlist is modified in place."},
	{"shift", shift, METH_O, "Execute the .shift() method on each segment in the list.  The algorithm is O(n) and does not require the list to be coalesced nor does it coalesce the list.  Segmentlist is modified in place."},
	{NULL,}
};


PyTypeObject segments_SegmentList_Type = {
	PyObject_HEAD_INIT(NULL)
	.tp_base = &PyList_Type,
	.tp_as_number = &as_number,
	.tp_as_sequence = &as_sequence,
	.tp_doc =
"The segmentlist class defines a list of segments, and is an\n" \
"extension of the built-in list class.  This class provides\n" \
"addtional methods that assist in the manipulation of lists of\n" \
"segments.  In particular, arithmetic operations such as union and\n" \
"intersection are provided.  Unlike the segment class, the\n" \
"segmentlist class is closed under all supported arithmetic\n" \
"operations.\n" \
"\n" \
"All standard Python sequence-like operations are supported, like\n" \
"slicing, iteration and so on, but the arithmetic and other methods\n" \
"in this class generally expect the segmentlist to be in what is\n" \
"refered to as a \"coalesced\" state --- consisting solely of disjoint\n" \
"segments listed in ascending order.  Using the standard Python\n" \
"sequence-like operations, a segmentlist can be easily constructed\n" \
"that is not in this state;  for example by simply appending a\n" \
"segment to the end of the list that overlaps some other segment\n" \
"already in the list.  The class provides a coalesce() method that\n" \
"can be called to put it in the coalesced state.  Following\n" \
"application of the coalesce method, all arithmetic operations will\n" \
"function reliably.  All arithmetic methods themselves return\n" \
"coalesced results, so there is never a need to call the coalesce\n" \
"method when manipulating segmentlists exclusively via the\n" \
"arithmetic operators.\n" \
"\n" \
"Example:\n" \
"\n" \
">>> x = segmentlist([segment(-10, 10)])\n" \
">>> x |= segmentlist([segment(20, 30)])\n" \
">>> x -= segmentlist([segment(-5, 5)])\n" \
">>> print x\n" \
"[segment(-10, -5), segment(5, 10), segment(20, 30)]\n" \
">>> print ~x\n" \
"[segment(-infinity, -10), segment(-5, 5), segment(10, 20), segment(30, infinity)]",
	/* FIXME: with Py_TPFLAGS_CHECKTYPES off, Python guarantees the
	 * arguments to all as_number methods will be segmentlist objects
	 * in which case some performance improvements could be realized by
	 * removing some type checking from these methods.  on the other
	 * hand, by adding just a bit more safety checking we could turn
	 * Py_TPFLAGS_CHECKTYPES on and allow greater flexibility in the
	 * use of this code. */
	.tp_flags = Py_TPFLAGS_DEFAULT /*| Py_TPFLAGS_CHECKTYPES*/ | Py_TPFLAGS_BASETYPE,
	.tp_methods = methods,
	.tp_name = MODULE_NAME ".SegmentList",
};
