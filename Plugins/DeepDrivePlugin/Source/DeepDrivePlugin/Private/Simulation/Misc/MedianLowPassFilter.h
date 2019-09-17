#pragma once

#include "CoreMinimal.h"

template<int32 FILTERWIDTH>
class TMedianLowPassFilter
{
public:

	TMedianLowPassFilter()
	{
		for(int32 i = 0; i < FILTERWIDTH; m_Buffer[i++] = 0.0f);
	}

	float add(float value)
	{
		m_Buffer[m_nextIndex] = value;

		m_nextIndex = (m_nextIndex + 1) % FILTERWIDTH;

		float sorted[FILTERWIDTH];
		for (int32 i = 0; i < FILTERWIDTH; sorted[i] = m_Buffer[i], i++);
		sort(sorted);

		return sorted[FILTERWIDTH / 2];
	}

private:

	void sort(float *values)
	{
		int32 i = 1;
		while(i < FILTERWIDTH)
		{
			float x = values[i];
			int32 j = i - 1;
			while(j >= 0 && values[j] > x)
			{
				values[j + 1] = values[j];
				j--;
			}
			values[j + 1] = x;
			i++;
		}
	}

	float			m_Buffer[FILTERWIDTH];
	uint32			m_nextIndex = 0;

};
