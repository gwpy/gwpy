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
 *                     Segments Module Component --- Main
 *
 * ============================================================================
 */

#include <Python.h>
#include <segments.h>


/*
 * ============================================================================
 *
 *                           Module Initialization
 *
 * ============================================================================
 */


void init__segments(void)
{
	/*
	 * Initialize module
	 */

	PyObject *module = Py_InitModule3(MODULE_NAME, NULL, "C implementations of the infinity, segment, and segmentlist classes from the segments module.");

	/*
	 * Create infinity class
	 */

	if(PyType_Ready(&segments_Infinity_Type) < 0)
		return;
	Py_INCREF(&segments_Infinity_Type);
	PyModule_AddObject(module, "infinity", (PyObject *) &segments_Infinity_Type);

	/*
	 * Create positive and negative infinity instances
	 */

	segments_PosInfinity = (segments_Infinity *) _PyObject_New(&segments_Infinity_Type);
	segments_NegInfinity = (segments_Infinity *) _PyObject_New(&segments_Infinity_Type);
	Py_INCREF(segments_PosInfinity);
	Py_INCREF(segments_NegInfinity);
	PyModule_AddObject(module, "PosInfinity", (PyObject *) segments_PosInfinity);
	PyModule_AddObject(module, "NegInfinity", (PyObject *) segments_NegInfinity);

	/*
	 * Create segment class.  Ideally the .tp_hash field would be
	 * initialized along with the other fields in the initializer in
	 * segment.c, but something about PyTuple_Type makes the compiler
	 * unhappy with that.
	 */

	if(!segments_Segment_Type.tp_hash)
		segments_Segment_Type.tp_hash = PyTuple_Type.tp_hash;
	if(PyType_Ready(&segments_Segment_Type) < 0)
		return;
	Py_INCREF(&segments_Segment_Type);
	PyModule_AddObject(module, "Segment", (PyObject *) &segments_Segment_Type);
	/* uninherit tp_print from tuple class */
	segments_Segment_Type.tp_print = NULL;

	/*
	 * Create segmentlist class
	 */

	if(PyType_Ready(&segments_SegmentList_Type) < 0)
		return;
	Py_INCREF(&segments_SegmentList_Type);
	PyModule_AddObject(module, "SegmentList", (PyObject *) &segments_SegmentList_Type);
}
