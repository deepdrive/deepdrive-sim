
#pragma once

#include "Runtime/Core/Public/Containers/UnrealString.h"
#include "Runtime/Core/Public/Containers/ContainersFwd.h"

template <class CTX, class STATE>
class TStateMachine
{

	typedef TStateMachine<CTX, STATE> StateMachine;
	typedef STATE StateBase;


public:

	TStateMachine()
		{	}

	void registerState(StateBase *state)
	{
		if(state)
			m_States.Add(state->getName(), state);
	}

	void update(CTX &ctx, float dT)
	{
		if(m_nextState)
		{
			if(m_nextStateOverride)
			{
				m_nextState = m_nextStateOverride;
				m_nextStateOverride = 0;
			}

			if(m_curState)
				m_curState->exit(ctx);

			m_prevState = m_curState;

			m_curState = m_nextState;
			m_nextState = 0;
			m_curState->enter(ctx);
		}
		else
		{
			if(m_curState)
				m_curState->update(ctx, dT);
		}
	}

	StateBase* setNextState(const FString &name)
	{
		StateBase *state = m_States.Contains(name) ? m_States[name] : 0;

		if(state && state != m_curState)
			setNextState(m_States[name]);
		return state;
	}


	void setNextState(StateBase *state)
	{
		m_nextState = state;
	}

	void overrideNextState(const FString &name)
	{
		StateBase *state = m_States.Contains(name) ? m_States[name] : 0;

		if(state && state != m_curState)
		{
			m_nextStateOverride = state;
		}
	}

	void revertToPreviousState()
	{
		m_nextState = m_prevState;
	}

protected:

	typedef TMap<FString, StateBase*> States;

	States			m_States;

	StateBase		*m_curState = 0;
	StateBase		*m_nextState = 0;
	StateBase		*m_prevState = 0;

	StateBase		*m_nextStateOverride = 0;
};

