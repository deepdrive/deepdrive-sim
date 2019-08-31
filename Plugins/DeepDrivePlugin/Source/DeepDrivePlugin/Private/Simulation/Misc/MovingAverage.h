#pragma once

#include "CoreMinimal.h"

template <int32 WIDTH>
class TMovingAverage
{
public:

	TMovingAverage()
	{
		for (int32 i = 0; i < WIDTH; m_Buffer[i++] = 0.0f)
			;
	}

	float add(float value)
	{
		m_curSum -= m_Buffer[m_nextIndex];
		m_Buffer[m_nextIndex] = value;
		m_nextIndex = (m_nextIndex + 1) % WIDTH;
		m_curSum += value;

		return m_curSum / static_cast<float> (WIDTH);
	}

private:

	float 	m_Buffer[WIDTH];
	int32 	m_nextIndex = 0;

	float	m_curSum = 0.0f;

};
