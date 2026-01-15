---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: posts
title: Safe Deep Deterministic Policy Gradient
permalink: /projects/sddpg/
description: Le vent se lève! Il faut tenter de vivre
classes: wide
author_profile: true
---

{::nomarkdown}
{% assign jupyter_path = "assets/demos/safe_rl/website.ipynb" | relative_url %}
{% capture notebook_exists %}{% {{ site.baseurl }}/assets/demos/safe_rl/website.ipynb %}{% endcapture %}
{% if notebook_exists == "true" %}
    {% jupyter_notebook jupyter_path %}
{% else %}
    <p>Sorry, the notebook you are looking for does not exist.</p>
{% endif %}
{:/nomarkdown}