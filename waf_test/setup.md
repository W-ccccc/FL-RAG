本实验采用 ModSecurity + OWASP Core Rule Set CRS 构建 Web 应用防火墙。运行载体为 WSL Ubuntu 环境中的 Apache2 + libapache2-mod-security2，
<br>

ModSecurity 工作模式设置为拦截模式：

SecRuleEngine On
<br>
即当请求命中高风险规则并达到 CRS 异常评分阈值时，WAF 直接返回阻断响应。
<br>
规则集
<br>
当前实际使用的是 Ubuntu 软件源自带的 OWASP CRS 规则集，路径为：
/usr/share/modsecurity-crs/rules/
该规则集包含约 31 个 .conf 规则文件，覆盖常见 Web 攻击类型，包括：
<br>
SQL 注入：REQUEST-942-APPLICATION-ATTACK-SQLI.conf
XSS：REQUEST-941-APPLICATION-ATTACK-XSS.conf
LFI：REQUEST-930-APPLICATION-ATTACK-LFI.conf
RCE：REQUEST-932-APPLICATION-ATTACK-RCE.conf
协议异常：REQUEST-920-PROTOCOL-ENFORCEMENT.conf
协议攻击：REQUEST-921-PROTOCOL-ATTACK.conf
扫描器检测：REQUEST-913-SCANNER-DETECTION.conf
方法限制：REQUEST-911-METHOD-ENFORCEMENT.conf
阻断评估：REQUEST-949-BLOCKING-EVALUATION.conf
