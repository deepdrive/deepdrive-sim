

#include "DeepDrivePluginPrivatePCH.h"
#include "PIDController.h"


PIDController::PIDController(float kp, float ki, float kd)
	:	m_Kp(kp)
	,	m_Ki(ki)
	,	m_Kd(kd)
	,	m_prevE(0.0f)
	,	m_SumE(0.0f)
{
	for (signed i = 0; i < HistoryLength; ++i)
		m_History[i] = 0.0f;
}

float PIDController::advance(float dT, float curE)
{
	const float dE = (curE - m_prevE);

	m_SumE -= m_History[m_lastHistoryIndex];
	m_SumE += curE;
	m_History[m_nextHistoryIndex] = curE;
	m_nextHistoryIndex = (m_nextHistoryIndex + 1) % HistoryLength;
	m_lastHistoryIndex = (m_lastHistoryIndex + 1) % HistoryLength;

	float y = m_Kp * curE + m_Ki * dT * m_SumE / static_cast<float> (HistoryLength) + dE * m_Kd / dT;

	m_prevE = curE;

	return y;
}
