### Solution Decomposition

#### Intent

The view defines the runtime decomposition of the server-side part of the solution. It is driven by the [Business Case](#business-case) and the architecture best practices applicable to the cloud-based applications.

#### Context

The view context is defined by the view [Solution Context](#solution-context) where this section represents the decomposition of the system using the Containers view.

#### Representation

[(add your container diagram below)]{.mark}

![Cloud Solution Container view](solution-decomposition.png){#fig:solution-decomposition width=96%}

[(explain your container diagram)]{.mark}

The cloud based part of the solution is decomposed into several parts documented below combining several standard architectural patterns applicable to the highly loaded cloud based SaaS applications: Load Balancer, Data Cache, Background Process, Shared Storage, and Static Content Provider. The subsection resoning provides detailed discussion of these choices.

#### Element Catalog

Table of annotated elements.

[(annotate elements from the container diagram here)]{.mark}

+----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| \#                                                                               | Description                                                                                   |
+:=================================================================================+===============================================================================================+
| Element1                                                                         | Responsible for a, b, c                                                                       |
+----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Element2                                                                         | Responsible for a, b, c                                                                       |
+----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| Element3                                                                         | Responsible for a, b, c                                                                       |
+----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
| \<element name\>                                                                 | \<element description and responsibilities\>                                                  |
+----------------------------------------------------------------------------------+-----------------------------------------------------------------------------------------------+
