## Key Risks and Open Issues

The section lists the key risks related to the solution design and implementation. It also lists key open issues where architectural decisions have not been made yet or are likely to change.

+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| \#         | Risk Description                                               | Mitigation Strategy                                            |
+:===========+================================================================+================================================================+
| R-1        | []{#risk-r-1}Custodian data is usually in the Microsoft 365    | Confirm the data sources in Phase 1. Design connectors for     |
|            | tenants of the clients, not in the firm's tenant.              | many client tenants.                                           |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-2        | []{#risk-r-2}Open-source libraries possibly cannot read some   | Import these sources as exports from forensic tools. Record    |
|            | rare formats, such as Lotus Notes archives and phone images.   | the hash value and metadata of each item.                      |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-3        | []{#risk-r-3}TAR can miss its recall and precision targets.    | Keep linear review available. Agree on the validation protocol |
|            | Court acceptance of TAR is different in each jurisdiction.     | with the court.                                                |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-4        | []{#risk-r-4}Privilege detection can miss privileged           | A reviewer makes the final decision. Review a sample of the    |
|            | documents. These documents then go to the opposing party.      | unflagged documents before production.                         |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-5        | []{#risk-r-5}The limits of 16 months, EUR 6 million and 12     | Deploy all regions from one infrastructure-as-code template.   |
|            | people are tight for 3 phases and 3 regions.                   | Give priority to Phases 1 and 2.                               |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-6        | []{#risk-r-6}Model validation is possible only with real       | Get the agreement of these clients in Phase 1. Use the         |
|            | matters from 3 large clients in the first year.                | migrated matters as more validation data.                      |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-7        | []{#risk-r-7}Client data cannot go on the platform before the  | Design the security controls from the start. Do a              |
|            | security assessment passes. A late failure moves go-live.      | pre-assessment before the formal assessment.                   |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
| R-8        | []{#risk-r-8}If go-live is late, the vendor contracts end      | Align go-live with the 2027 contract renewals. Negotiate short |
|            | before the platform is ready.                                  | extensions as a fallback.                                      |
+------------+----------------------------------------------------------------+----------------------------------------------------------------+
