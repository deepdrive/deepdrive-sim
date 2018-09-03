

#pragma once

#include "CoreMinimal.h"


/**
 * 
 */
class PIDController
{
public:

	PIDController(float kp, float ki, float kd);

	void reset();

	float advance(float dT, float curE);

private:

	float			m_Kp;
	float			m_Ki;
	float			m_Kd;

	float			m_prevE;
	float			m_SumE;

	enum
	{
		HistoryLength = 10
	};

	float			m_History[HistoryLength];
	int32			m_nextHistoryIndex = 0;
};
