#pragma once
#pragma warning(push, 0)
#ifdef _WIN64
#include <aclapi.h>
#include <dxgi1_2.h>
#include <windows.h>
#include <VersionHelpers.h>
#define _USE_MATH_DEFINES
#endif

#ifdef _WIN64
class WindowsSecurityAttributes
{
protected:
    SECURITY_ATTRIBUTES mWinSecurityAttributes;
    PSECURITY_DESCRIPTOR mWinPSecurityDescriptor;

public:
    WindowsSecurityAttributes();
    SECURITY_ATTRIBUTES* operator&();
    ~WindowsSecurityAttributes();
};
#endif
#pragma warning(pop)