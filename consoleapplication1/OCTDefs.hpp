/*-------------------------------------------------------------------*/
/*                                                                   */
/*                    Zeiss Interventional Imaging                   */
/*                         Research Solution                         */
/*                   -----------------------------                   */
/*  Chair for Computer Aided Medical Procedures & Augmented Reality  */
/*                  Technische Universität München                   */
/*                                                                   */
/*-------------------------------------------------------------------*/

#pragma once
#ifndef __FRAMEOCTDEFS
#define __FRAMEOCTDEFS

#include <string>

/**
* \brief The scan type of the OCT device
*/
enum ScanType
{
	NONE = 0,
	ONE = 1,
	CROSS = 2,
	FIVE = 5,
	RADIAL = 6,
	PARALLEL10 = 10,
	GRID7X7 = 14,
	PARALLEL15 = 15,
	GRID9X9 = 18,
	GRID11X11 = 22,
	GRID13X13 = 26,
	PARALLEL30 = 30,
	CUBE128 = 128,
	CUBE200 = 200
};

/**
* \brief Structure for holding information regarding a recorded OCT frame
*/
struct PatternData
{
	std::string fileName;
	int fileNumber;
	int scanNumber;
	ScanType scanType;
	int pixelWidth;
	int pixelHeight;
	int patternWidth;
	int patternHeight;
	double patternOffsetX;
	double patternOffsetY;
	int patternRotation;
	double patternScaleX;
	double patternScaleY;
	long long patternTimestamp;
	int referenceArmPosition;
	bool edi;
};

inline int OCTFrameSortProc(const PatternData &elem1, const PatternData &elem2)
{
	return elem1.fileNumber < elem2.fileNumber;
}

#endif