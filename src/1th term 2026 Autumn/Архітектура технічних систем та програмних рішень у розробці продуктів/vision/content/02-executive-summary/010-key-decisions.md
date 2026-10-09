## Key Decisions

The section outlines key design decisions about the solution including the architecture big picture and most essential technologies and external services to rely on.

The solution will be a firm-operated eDiscovery platform on Amazon Web Services (AWS). The key decisions are:

- **Cloud provider.** The platform uses AWS, because the delivery team has strong AWS experience. This decreases the delivery risk in the 16-month program. The platform connects to Microsoft 365 and Purview through the Microsoft Graph API. The Client's Microsoft Entra ID gives single sign-on (SSO) with multi-factor authentication (MFA).
- **Data residency.** Each region (EU, UK and US) has an independent platform instance in its own AWS accounts. Service control policies block all other regions. Thus, the data of a matter stays in the region that its client approves. A separate AWS KMS key encrypts the data of each client.
- **Build approach.** The team develops the eDiscovery workflow. AWS managed services and open-source libraries do the generic functions: Apache Tika extracts text and metadata, Amazon Textract does optical character recognition (OCR), and Amazon OpenSearch Service does the full-text search. Thus, there are no license costs and no new vendor dependency. Sources without a standard connector, such as mobile devices, come in as exports from forensic tools.
- **Processing and storage.** Amazon SQS, AWS Step Functions and Amazon ECS with AWS Fargate process the collected data. The number of workers increases to process 2 TB of raw data each day. Amazon S3 keeps the documents, and Amazon Aurora PostgreSQL keeps the matter data.
- **Review.** Technology-assisted review (TAR) uses continuous active learning and statistical validation on samples. Courts in the UK, the US and some EU jurisdictions accept this method. Large language models in Amazon Bedrock help with privilege detection, through endpoints in the same region only. A reviewer always makes the final privilege decision.
- **Audit.** An append-only, hash-chained ledger records all user and system actions. Copies of its digests go to S3 Object Lock storage. Thus, any change to the audit trail is visible.
