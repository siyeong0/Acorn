#include "WindowsSecurityAttributes.h"

#pragma warning(disable : 6011 6387)

#ifdef _WIN64
WindowsSecurityAttributes::WindowsSecurityAttributes()
{
	mWinPSecurityDescriptor = (PSECURITY_DESCRIPTOR)calloc(1, SECURITY_DESCRIPTOR_MIN_LENGTH + 2 * sizeof(void**));

	PSID* ppSID = (PSID*)((PBYTE)mWinPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	InitializeSecurityDescriptor(mWinPSecurityDescriptor, SECURITY_DESCRIPTOR_REVISION);

	SID_IDENTIFIER_AUTHORITY sidIdentifierAuthority = SECURITY_WORLD_SID_AUTHORITY;
	AllocateAndInitializeSid(&sidIdentifierAuthority, 1, SECURITY_WORLD_RID, 0, 0, 0, 0, 0, 0, 0, ppSID);

	EXPLICIT_ACCESS explicitAccess;
	ZeroMemory(&explicitAccess, sizeof(EXPLICIT_ACCESS));
	explicitAccess.grfAccessPermissions = STANDARD_RIGHTS_ALL | SPECIFIC_RIGHTS_ALL;
	explicitAccess.grfAccessMode = SET_ACCESS;
	explicitAccess.grfInheritance = INHERIT_ONLY;
	explicitAccess.Trustee.TrusteeForm = TRUSTEE_IS_SID;
	explicitAccess.Trustee.TrusteeType = TRUSTEE_IS_WELL_KNOWN_GROUP;
	explicitAccess.Trustee.ptstrName = (LPTSTR)*ppSID;

	SetEntriesInAcl(1, &explicitAccess, NULL, ppACL);

	SetSecurityDescriptorDacl(mWinPSecurityDescriptor, TRUE, *ppACL, FALSE);

	mWinSecurityAttributes.nLength = sizeof(mWinSecurityAttributes);
	mWinSecurityAttributes.lpSecurityDescriptor = mWinPSecurityDescriptor;
	mWinSecurityAttributes.bInheritHandle = TRUE;
}

SECURITY_ATTRIBUTES* WindowsSecurityAttributes::operator&()
{
	return &mWinSecurityAttributes;
}

WindowsSecurityAttributes::~WindowsSecurityAttributes()
{
	PSID* ppSID = (PSID*)((PBYTE)mWinPSecurityDescriptor + SECURITY_DESCRIPTOR_MIN_LENGTH);
	PACL* ppACL = (PACL*)((PBYTE)ppSID + sizeof(PSID*));

	if (*ppSID)
	{
		FreeSid(*ppSID);
	}
	if (*ppACL)
	{
		LocalFree(*ppACL);
	}
	free(mWinPSecurityDescriptor);
}
#endif

#pragma warning(default : 6011 6387)