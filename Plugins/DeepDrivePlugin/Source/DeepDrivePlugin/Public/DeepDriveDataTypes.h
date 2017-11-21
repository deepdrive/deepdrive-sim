
#pragma once

#include "Engine.h"

struct DeepDriveVector2
{
	DeepDriveVector2(double _x = 0.0, double _y = 0.0)
		:	x(_x)
		,	y(_y)
	{	}

	double	x;
	double	y;
};

struct DeepDriveVector3
{

	DeepDriveVector3(const FVector &vec)
		:	x(vec.X)
		,	y(vec.Y)
		,	z(vec.Z)
	{	}

	DeepDriveVector3(double _x = 0.0, double _y = 0.0, double _z = 0.0)
		:	x(_x)
		,	y(_y)
		,	z(_z)
	{	}

	double	x;
	double	y;
	double	z;
};

struct DeepDriveVector4
{
	DeepDriveVector4(double _x = 0.0, double _y = 0.0, double _z = 0.0, double _w = 0.0)
		:	x(_x)
		,	y(_y)
		,	z(_z)
		,	w(_w)
	{	}

	double	x;
	double	y;
	double	z;
	double	w;
};
