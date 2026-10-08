# Case 50: eDiscovery and Litigation Document Review Platform (Legal Services)

Extracted from `IT_Project_RFP_Business_Cases_Generation_2.docx`, the course's set of 68 RFP business cases. The text below is reproduced verbatim; only the heading levels are adjusted for a standalone file.

| Case ID | Project Domain | Project Theme | Key Strategic Technology | Primary Business Metric (KPI) |
| :---- | :---- | :---- | :---- | :---- |
| Case 50 | Legal Services | eDiscovery and Litigation Document Review Platform | NLP / Technology-Assisted Review | Reduce document review cost by 45% |

## Executive Summary

**Business Case Summary**

The organization is an international law firm with 1,400 lawyers in 14 offices whose litigation, arbitration and regulatory-investigation practices process roughly 220 million documents per year across 300 active matters, the largest of which exceed 20 million documents. Review is currently outsourced to three hosting vendors and staffed with contract reviewers, costing the firm and its clients EUR 18.5 million per year. Reviewers average 52 documents per hour, 68 percent of matters exceed their review budget, and legal-hold notices take an average of five business days to issue because custodians are identified manually. The proposed initiative establishes a firm-operated eDiscovery platform covering legal hold, collection, processing, analytics, technology-assisted review with continuous active learning, privilege detection and production, hosted under the firm's own security controls. Based on comparable deployments, the firm expects review cost to fall by 45 percent, equivalent to about EUR 8.3 million per year, the share of documents needing human review to fall by 60 percent, and legal-hold issuance to drop to under four hours. Strategically, the platform lets the firm offer fixed-fee litigation services, meet client demands for demonstrable data security, and retain review margin that currently goes to external vendors.

**Anticipated Outcomes**

* Reduce total document review cost by 45 percent, about EUR 8.3 million per year, within 24 months of the platform going live.

* Reduce the proportion of collected documents requiring eyes-on human review from 100 percent to 40 percent through technology-assisted review on matters above 500,000 documents.

* Cut legal-hold issuance time from five business days to under four hours and achieve 100 percent tracked custodian acknowledgements within 18 months.

* Host 90 percent of new matters on the in-house platform by month 20, ending dependence on external hosting vendors for standard litigation work.

**Critical Success Factors**

* The technology-assisted review workflow must reach at least 75 percent recall at 65 percent or better precision on validation samples, with statistically defensible measurement acceptable to courts in the firm's main jurisdictions.

* Privilege-detection models must identify at least 95 percent of privileged documents in sampled sets so that inadvertent production of privileged material is eliminated on matters using the platform.

* The platform must pass an independent ISO 27001-aligned security assessment and the client security questionnaires of the firm's ten largest clients before any client data is hosted.

## Business Need

**Business Context**

Litigation and regulatory investigations increasingly turn on electronically stored information from email, chat platforms, mobile devices and cloud collaboration tools, and data volumes per matter have tripled in five years. Courts in the United Kingdom, the United States and several EU jurisdictions now accept and in some cases expect technology-assisted review, while clients demand fixed fees, cost transparency and evidence that their confidential data is protected. Competing firms have brought eDiscovery in-house to capture margin and shorten timelines, and the firm risks losing panel appointments if its review costs remain vendor-driven.

**Problem Statement**

Review work is scattered across three vendor platforms with different workflows, so partners cannot compare progress or cost across matters, and each vendor onboarding adds two to three weeks before review begins. Contract reviewers work linearly through document sets, averaging 52 documents per hour, and 68 percent of matters overran their review budget last year, with write-offs of EUR 2.1 million. Legal-hold notices are issued by email and tracked in spreadsheets, producing acknowledgement gaps that have been challenged in two recent proceedings. Client data moves between vendors on encrypted drives, which several clients have flagged as unacceptable in security audits.

**Business Need**

The firm requires a single, secure capability to place and track legal holds, collect data from corporate and cloud sources, process and de-duplicate it, prioritise review with machine learning, protect privileged content, and produce documents in court-ready formats. The capability must give partners real-time visibility of cost and progress per matter, support reviewers in any office, and demonstrate to clients and courts that the review process is defensible, auditable and confined to jurisdictions the client approves.

## Recommended Solution

**Solution Overview**

The recommended solution is a firm-operated eDiscovery platform delivered in three phases. Phase one implements legal hold and collection connectors together with processing and early case assessment. Phase two adds the review workspace with continuous active learning, privilege analytics, redaction and production. Phase three delivers client reporting portals and advanced analytics such as communication mapping. Data flows from custodian sources into a processing tier, then into matter-specific review workspaces, with all events recorded in an audit ledger and cost data exported to the firm's billing system.

**Solution Detail**

User roles include litigation partners, review managers, contract reviewers, eDiscovery analysts, security administrators and client counsel. Clients are a browser-based review workspace, a tablet view for partners and APIs for bulk import and export. The platform must integrate with Microsoft 365 and Purview for preservation and collection, the firm's iManage Work document management system, the practice-management and billing system, and the matter-intake and conflicts system for ethical walls. Non-functional targets: ingest 2 TB of raw data per day, host matters of 20 million documents, support 500 concurrent reviewers, return full-matter searches within three seconds, 99.9 percent availability, RPO 15 minutes and RTO four hours, with data kept in the client-approved region (EU, UK or US). Security requires SSO with MFA, role-based access with matter-level ethical walls, per-client encryption keys, a complete audit trail and defensible deletion at matter close. Administrators need dashboards for capacity, cost and review throughput.

**Assumptions and Limitations**

* **Assumptions:** It is assumed that the firm's Microsoft 365 tenant and document management system expose supported connectors for preservation and collection, and that at least three large clients will agree to move ongoing matters to the platform during the first year to provide validation data. It is further assumed that the litigation support team of eight analysts will be retained and retrained as platform operators, and that existing vendor contracts can be wound down at their 2027 renewal points without penalty.

* **Limitations:** The programme is capped at 16 months and EUR 6 million, including migration of ten historical matters, with a core delivery team of no more than twelve people. Some legacy matters held by vendors in proprietary formats may only be migrated as static productions, losing coding history. Court acceptance of technology-assisted review varies by jurisdiction, so linear review must remain available on request. Client data-residency rules may prevent a single hosting footprint and require region-specific instances from the outset.
