## Design Constraints

The section lists the constraints accounted for in the designed solution. These can be of business, technical, resource, and other types.

+----------------+----------------------------------------------------------------------------------------------------+
| \#             | Description                                                                                        |
+================+====================================================================================================+
| CONST-1        | []{#constraint-const-1}**Business.** The program must take no more than 16 months.                 |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-2        | []{#constraint-const-2}**Business.** The program must cost no more than EUR 6 million, with the    |
|                | migration of 10 historical matters included.                                                       |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-3        | []{#constraint-const-3}**Business.** The platform must be ready for the end of the vendor          |
|                | contracts at their 2027 renewal points.                                                            |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-4        | []{#constraint-const-4}**Resource.** The core delivery team must have no more than 12 persons.     |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-5        | []{#constraint-const-5}**Resource.** The 8 analysts of the litigation support team must operate    |
|                | the platform, after retraining.                                                                    |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-6        | []{#constraint-const-6}**Legal.** The data of a matter must stay in the region that its client     |
|                | approves: the EU, the UK or the US. Thus, regional instances are necessary from the start.         |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-7        | []{#constraint-const-7}**Legal.** Linear review must stay available on request, because court      |
|                | acceptance of TAR is different in each jurisdiction.                                               |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-8        | []{#constraint-const-8}**Security.** The platform must pass an independent ISO 27001-aligned       |
|                | security assessment and the security questionnaires of the 10 largest clients before it keeps      |
|                | client data.                                                                                       |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-9        | []{#constraint-const-9}**Security.** A separate encryption key must protect the data of each       |
|                | client.                                                                                            |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-10       | []{#constraint-const-10}**Security.** Users must sign in through SSO with MFA.                     |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-11       | []{#constraint-const-11}**Technical.** The platform must integrate with the existing systems of    |
|                | the Client: Microsoft 365 and Purview, iManage Work, the practice-management and billing system,   |
|                | and the matter-intake and conflicts system.                                                        |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-12       | []{#constraint-const-12}**Technical.** The platform must use AWS, because the delivery team has    |
|                | strong AWS experience.                                                                             |
+----------------+----------------------------------------------------------------------------------------------------+
| CONST-13       | []{#constraint-const-13}**Technical.** Some historical matters are in proprietary vendor formats.  |
|                | These matters can come to the platform only as static productions, without their coding history.   |
+----------------+----------------------------------------------------------------------------------------------------+

[CONST-3](#constraint-const-3) and [CONST-12](#constraint-const-12) are soft constraints. The Client can negotiate short extensions with the vendors ([R-8](#risk-r-8)). AWS is a decision of the delivery team, not a requirement of the Client. All other constraints are hard.
