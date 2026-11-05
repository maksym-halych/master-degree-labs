### Use Case View eDiscovery Lifecycle

#### View Context

The interactions in this view happen between the users of the platform, its external systems and the platform. They cover the lifecycle of a matter, from its registration to its deletion, and the features [F-1](#feature-f-1) to [F-10](#feature-f-10).

#### Representation

![Legal hold, collection and processing use cases](use-cases-collection.pdf){#fig:use-cases-collection width=90%}

@fig:use-cases-collection shows the features [F-1](#feature-f-1), [F-2](#feature-f-2) and [F-3](#feature-f-3) of Phase 1. It also shows the registration of a matter, which is part of [F-7](#feature-f-7). The eDiscovery analyst preserves, collects and processes the data of the matter.

![Review and production use cases](use-cases-review.pdf){#fig:use-cases-review width=82%}

@fig:use-cases-review shows the features [F-4](#feature-f-4), [F-5](#feature-f-5), [F-6](#feature-f-6) and [F-10](#feature-f-10) of Phase 2. The review manager and the contract reviewers examine the documents. The eDiscovery analyst produces the result and migrates the historical matters.

![Matter security use cases](use-cases-security.pdf){#fig:use-cases-security width=80%}

@fig:use-cases-security shows the feature [F-7](#feature-f-7), which applies to all phases. All users sign in through one identity provider. The security administrator controls each matter until its deletion.

![Monitoring and reporting use cases](use-cases-monitoring.pdf){#fig:use-cases-monitoring width=87%}

@fig:use-cases-monitoring shows the feature [F-8](#feature-f-8) of Phases 1 and 2, and the feature [F-9](#feature-f-9) of Phase 3.

This view defines primary Use Cases for the features [F-1](#feature-f-1) to [F-10](#feature-f-10), corresponding to the types of interaction between the users, the external systems and the solution.

#### Element Catalog

+------------+--------------------------------+------------------------------------------------------------------------------+
| \#         | Name                           | Description                                                                  |
+============+================================+==============================================================================+
| ACT-1      | Litigation partner             | Leads a matter. Monitors its cost and progress.                              |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-2      | Review manager                 | Sets up the review of a matter and controls its quality.                     |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-3      | Contract reviewer              | Reads documents and records their coding.                                    |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-4      | eDiscovery analyst             | Operates the platform. Does legal holds, collection, processing and          |
|            |                                | production.                                                                  |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-5      | Security administrator         | Controls access, ethical walls, encryption keys and the audit trail.         |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-6      | Client counsel                 | A lawyer of the client. Monitors the matters of this client.                 |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-7      | Platform user                  | Any person in the roles ACT-1 to ACT-6.                                      |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-8      | Custodian                      | Gets hold notices and acknowledges them.                                     |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-9      | Opposing party or court        | Gets the documents that the firm produces.                                   |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-10     | External hosting vendor        | Keeps historical matters today. Gives their data for migration.              |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-11     | Matter-intake and conflicts    | External system. Sends new matters and their ethical walls.                  |
|            | system                         |                                                                              |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-12     | Microsoft 365 and Purview      | External system. Preserves custodian data and gives it for collection.       |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-13     | iManage Work                   | External system. The document management system of the firm. Gives documents |
|            |                                | for collection and keeps copies of productions.                              |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-14     | Practice-management and        | External system. Gets the cost data of each matter.                          |
|            | billing system                 |                                                                              |
+------------+--------------------------------+------------------------------------------------------------------------------+
| ACT-15     | Microsoft Entra ID             | External system. Gives SSO with MFA.                                         |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-1       | []{#usecase-uc-1}Register a    | The matter-intake and conflicts system sends a new matter and its ethical    |
|            | matter                         | walls. The platform makes a workspace for the matter in the region that the  |
|            |                                | client approves. ([F-7](#feature-f-7))                                       |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-2       | []{#usecase-uc-2}Issue a legal | The eDiscovery analyst selects the custodians and sends the hold notices.    |
|            | hold                           | The platform preserves their data in Microsoft 365 and Purview.              |
|            |                                | ([F-1](#feature-f-1))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-3       | []{#usecase-uc-3}Acknowledge a | The custodian opens the hold notice and acknowledges it. The platform        |
|            | legal hold                     | records this action. ([F-1](#feature-f-1))                                   |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-4       | []{#usecase-uc-4}Track         | The eDiscovery analyst monitors the status of each custodian and sends       |
|            | acknowledgements               | reminders. ([F-1](#feature-f-1))                                             |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-5       | []{#usecase-uc-5}Collect       | The eDiscovery analyst starts a collection through a standard connector. The |
|            | custodian data                 | data comes from Microsoft 365, Purview or iManage Work.                      |
|            |                                | ([F-2](#feature-f-2))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-6       | []{#usecase-uc-6}Import data   | The eDiscovery analyst sends data from other sources, such as exports from   |
|            | in bulk                        | forensic tools, to the bulk import API. The platform records the hash value  |
|            |                                | and metadata of each item. ([F-2](#feature-f-2))                             |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-7       | []{#usecase-uc-7}Process       | The eDiscovery analyst starts the processing. The platform extracts text and |
|            | collected data                 | metadata, does OCR and removes duplicates. ([F-3](#feature-f-3))             |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-8       | []{#usecase-uc-8}Assess a      | The eDiscovery analyst and the review manager examine the volume and cost    |
|            | matter early                   | estimates. They remove irrelevant data before review. ([F-3](#feature-f-3))  |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-9       | []{#usecase-uc-9}Set up a      | The review manager selects TAR or linear review, and gives batches of        |
|            | review                         | documents to the contract reviewers. ([F-4](#feature-f-4))                   |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-10      | []{#usecase-uc-10}Review       | The contract reviewer reads each document and records its coding. With TAR,  |
|            | documents                      | the platform retrains its model on each decision and shows the most relevant |
|            |                                | documents first. ([F-4](#feature-f-4))                                       |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-11      | []{#usecase-uc-11}Validate TAR | The review manager selects a validation sample. The platform calculates      |
|            | results                        | recall and precision with a statistical method that courts accept.           |
|            |                                | ([F-4](#feature-f-4))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-12      | []{#usecase-uc-12}Decide on    | The platform flags documents that are possibly privileged. The contract      |
|            | privilege                      | reviewer makes the final decision. The review manager examines a sample of   |
|            |                                | the unflagged documents. ([F-5](#feature-f-5))                               |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-13      | []{#usecase-uc-13}Redact       | The contract reviewer masks privileged or sensitive content in a document.   |
|            | documents                      | ([F-5](#feature-f-5))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-14      | []{#usecase-uc-14}Produce      | The eDiscovery analyst selects the relevant, non-privileged documents and    |
|            | documents                      | converts them to a court-ready format. The platform sends them to the        |
|            |                                | opposing party or the court, and keeps a copy in iManage Work.               |
|            |                                | ([F-6](#feature-f-6))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-15      | []{#usecase-uc-15}Migrate a    | The eDiscovery analyst imports a historical matter from an external hosting  |
|            | historical matter              | vendor. A matter in a proprietary format comes in as a static production,    |
|            |                                | without its coding history. ([F-10](#feature-f-10))                          |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-16      | []{#usecase-uc-16}Sign in      | A platform user signs in through Microsoft Entra ID with SSO and MFA.        |
|            |                                | ([F-7](#feature-f-7))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-17      | []{#usecase-uc-17}Manage       | The security administrator gives roles to users and changes the ethical      |
|            | access and ethical walls       | walls of a matter. ([F-7](#feature-f-7))                                     |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-18      | []{#usecase-uc-18}Manage       | The security administrator makes, replaces and disables the encryption key   |
|            | client keys                    | of each client. ([F-7](#feature-f-7))                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-19      | []{#usecase-uc-19}Examine the  | The security administrator searches the audit trail and makes a report for a |
|            | audit trail                    | client or a court. The platform shows if a record changed.                   |
|            |                                | ([F-7](#feature-f-7))                                                        |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-20      | []{#usecase-uc-20}Delete a     | The security administrator starts the deletion at matter close. The platform |
|            | closed matter                  | deletes all data of the matter and records proof of the deletion. It does    |
|            |                                | not delete a matter with an active legal hold. ([F-7](#feature-f-7))         |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-21      | []{#usecase-uc-21}Monitor      | The litigation partner monitors the cost and progress of a matter in a       |
|            | matter cost and progress       | browser or on a tablet, and compares matters. ([F-8](#feature-f-8))          |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-22      | []{#usecase-uc-22}Export cost  | The platform sends the cost data of each matter to the practice-management   |
|            | data                           | and billing system. ([F-8](#feature-f-8))                                    |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-23      | []{#usecase-uc-23}Monitor      | The eDiscovery analyst monitors the capacity of the platform. The review     |
|            | capacity and throughput        | manager monitors the review throughput. ([F-8](#feature-f-8))                |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-24      | []{#usecase-uc-24}View client  | The client counsel monitors the cost and progress of the matters of this     |
|            | reports                        | client in the client reporting portal (Phase 3). ([F-9](#feature-f-9))       |
+------------+--------------------------------+------------------------------------------------------------------------------+
| UC-25      | []{#usecase-uc-25}Examine      | The eDiscovery analyst or the litigation partner examines a map of the       |
|            | communication maps             | communications between custodians (Phase 3). ([F-9](#feature-f-9))           |
+------------+--------------------------------+------------------------------------------------------------------------------+
