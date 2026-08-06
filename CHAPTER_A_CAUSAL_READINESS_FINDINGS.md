# Chapter A Causal Readiness Findings

Date: 2026-06-27

Chapter A is the causal what-if planning chapter.  It is intentionally
fail-closed.  The validated DKA/sepsis/AKI/respiratory routers are factual
forecasting artifacts; they do not identify treatment effects.

## Current Status

Observational EHR has already been useful for:

- factual forecasting;
- negative confounding diagnostics;
- target-router validation;
- data coverage audits.

It is not enough for Chapter A causal claims.  The current observational
diagnostics fail because of structural issues:

- poor treatment overlap;
- co-treatment and bundled ICU protocols;
- severity-driven treatment assignment;
- insufficient balance after adjustment.

This means the causal gate is not waiting for another router.  It is waiting for
external identification evidence.

## Candidate External Sources

The source registry is:

- BioLINCC: candidate NHLBI randomized-trial source, not in workspace
- Vivli: candidate randomized-trial source, not in workspace
- YODA: candidate randomized-trial source, not in workspace
- observational EHR: available for factual prediction and negative diagnostics
  only

Default readiness specs now cover four disease chapters:

- DKA insulin strategy RCT
- sepsis fluid strategy RCT
- AKI renal-support / RRT timing strategy RCT
- respiratory oxygenation / ventilation strategy RCT
- cardiovascular vasopressor / inotrope strategy RCT
- acute neuro hemodynamic / oxygenation strategy RCT
- hepatic failure resuscitation / support strategy RCT
- coagulopathy transfusion / anticoagulation strategy RCT

All specs fail closed until an external evidence table is mapped into the
contract.

## Required Data Contract

A candidate causal dataset must provide:

- patient or participant identifier;
- assignment or externally justified treatment instrument/front-door variable;
- treatment received/adherence column;
- baseline covariates measured before time zero;
- outcome measured after time zero;
- missingness support;
- enough subjects per arm or exposure strategy.

For randomized data, the readiness gate requires:

- adequate subject count;
- two-arm or explicitly modeled multi-arm assignment;
- acceptable outcome missingness;
- acceptable treatment adherence;
- acceptable baseline standardized mean differences.

For instrumental-variable or front-door designs, external review is required
before any causal claim can be enabled.

## Boundary

Passing Chapter A readiness would allow only a research causal-effect estimate.
It would not allow:

- clinical recommendation authority;
- runtime treatment action authority;
- checkpoint promotion;
- active symbolic-rule promotion.

Until an external dataset is mapped into `osler_jepa/causal_readiness.py`, all
causal claims remain closed.
