## Quality Attribute Scenarios

A Quality Attribute Scenario is an unambiguous and testable requirement for one or more Solution Quality Attributes such as Performance, Usability, Maintainability and others. The scenario consists of six parts: Source of Stimulus, Stimulus, Environment, Artifact, Response, testable and accurate Response Measure.

This section lists and prioritizes the scenarios pertinent to the designed solution.

[(List your quality attributes in the table below (at least 5))]{.mark}

+---------------------------------+------------------------------------+----------------------------------------+-------------------------------+---------------------------------+
| \#                              | Quality Attribute                  | Scenario                               | Business Priority             | Related                         |
|                                 |                                    |                                        |                               |                                 |
|                                 |                                    |                                        |                               | To                              |
+=================================+====================================+========================================+===============================+=================================+
| QA-1                            | Security                           | At all times the credentials entered   | High                          | [UC-1](#usecase-uc-1)           |
|                                 |                                    | by the user during log-in are          |                               |                                 |
|                                 | (credentials transport security)   | transferred to the server over         |                               |                                 |
|                                 |                                    | encrypted, secure channel without the  |                               |                                 |
|                                 |                                    | chance of sniffing by third party.     |                               |                                 |
+---------------------------------+------------------------------------+----------------------------------------+-------------------------------+---------------------------------+
| QA-2                            | Usability                          | When logged in and navigated to the    | Medium                        | [UC-2](#usecase-uc-2)           |
|                                 |                                    | payment page it takes the user up to 3 |                               |                                 |
|                                 | (easiness of payment)              | clicks to pay with the preregistered   |                               |                                 |
|                                 |                                    | valid credit card                      |                               |                                 |
+---------------------------------+------------------------------------+----------------------------------------+-------------------------------+---------------------------------+
| \<scenario id\>                 | \<attribute name                   | \<quality attribute scenario           | \<priority level: High,       | \<use case, feature, constraint |
|                                 |                                    | description\>                          | Medium, Low\>                 | ids\>                           |
|                                 | (scenario meaning)\>               |                                        |                               |                                 |
+---------------------------------+------------------------------------+----------------------------------------+-------------------------------+---------------------------------+

[(At least one QA from the table above should be documented in the graphic format as below)]{.mark}

**Scenario:** QA-1

**Quality Attributes:** Security

**Business Priority:** High

**Related To:** [UC-1](#usecase-uc-1)

**Description:** At all times the credentials entered by the user during log-in are transferred to the server over encrypted, secure channel without the chance of sniffing by third party.

**Environment:** Normal operation conditions

\begin{figure}[htbp]
\centering
\begin{tikzpicture}[
  font=\small,
  part/.style={draw, rectangle, text width=26mm, align=center, minimum height=17mm, inner sep=4pt},
  lbl/.style={font=\small\bfseries},
  flow/.style={-{Latex[length=2.5mm]}},
]
\node[part] (source) {Login Form};
\node[part, right=32mm of source] (artifact) {Client-Server Channel};
\node[part, right=32mm of artifact] (measure) {Traffic encrypted and protected from sniffing};

\node[lbl, above=1.5mm of source.north west, anchor=south west] (lsource) {Source};
\node[lbl, above=1.5mm of artifact.north west, anchor=south west] (lartifact) {Artifact};
\node[lbl, above=1.5mm of measure.north west, anchor=south west] (lmeasure) {Response Measure};

\draw[flow] (source) -- node[above, align=center, font=\footnotesize] {Enter and submit\\credentials} node[below, lbl] {Stimulus} (artifact);
\draw[flow] (artifact) -- node[above, align=center, font=\footnotesize] {Credentials\\transferred to server} node[below, lbl] {Response} (measure);

\node[above=9mm of artifact.north, anchor=south] (env) {\textbf{Environment}: The system is online and accessible from outside};

\node[draw, dashed, inner sep=4mm, fit=(source)(artifact)(measure)(env)(lsource)(lmeasure)] {};
\end{tikzpicture}
\caption{Quality attribute scenario QA-1 in graphic form}
\label{fig:qa-scenario}
\end{figure}
