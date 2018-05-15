
#pragma once

#include "Runtime/Core/Public/Containers/UnrealString.h"


template <class CTX>
class TStateBase
{
public:

	TStateBase(const FString &name)
		:	m_Name(name)
	{
	}

	virtual void enter(CTX &ctx)
	{
	}

	virtual void update(CTX &ctx, float dT)
	{
	}

	virtual void exit(CTX &ctx)
	{
	}

	const FString& getName() const
	{
		return m_Name;
	}

protected:

	FString				m_Name;

};

