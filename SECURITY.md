# Security Policy

HAMi-core (`libvgpu.so`) is the in-container enforcement component of
[HAMi](https://github.com/Project-HAMi/HAMi) and follows the
[HAMi security policy](https://github.com/Project-HAMi/HAMi/security/policy).

## Reporting a Vulnerability

Please **do not** open a public issue for a security report.

Report privately through
[HAMi's GitHub Security Advisories](https://github.com/Project-HAMi/HAMi/security/advisories/new)
and mention HAMi-core in the title. Reports are triaged by the maintainers
listed in [OWNERS](OWNERS).

Please include:

- A clear description of the vulnerability.
- Steps to reproduce, ideally the CUDA or NVML calls involved.
- The impact, and which tenant or process is affected.
- A suggested fix, if you have one.

## Supported Versions

HAMi-core carries no tags of its own. It is consumed as the `libvgpu`
submodule of [HAMi](https://github.com/Project-HAMi/HAMi) and released on
HAMi's version line, as the `projecthami/hamicore` image and inside
`projecthami/hami`. Fixes land on `main` and reach users with the next HAMi
release, so the HAMi security policy is what lists which versions receive
security fixes.

## Scope

HAMi-core limits GPU memory and compute for cooperative multi-tenant sharing on
a trusted cluster. It is not a hard security boundary against a workload with
enough privilege to bypass its own hook, for example by unsetting `LD_PRELOAD`,
using a statically linked binary, or `ptrace`.

- A workload exceeding its own quota, without affecting another tenant, is a
  correctness bug. Please open a normal issue for it.
- A workload reaching another tenant's memory, device or namespace that it was
  not granted is a vulnerability. Please report it privately.

## Response Process

Maintainers aim to acknowledge a report within 5 working days. Weekends,
holidays and time zone differences can affect that.
