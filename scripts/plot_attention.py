#
# visualize attention with heatmaps
#
# - requires seaborn and pandas
# - attention needs to be converted to cvs files
#   (see attention2cvs.pl)
# - TODO: replace hard-coded output file ('attention.pdf')


import sys
import seaborn as sns; sns.set()
import pandas as pd
import re

# http://seaborn.pydata.org/generated/seaborn.heatmap.html

data = pd.read_csv(sys.argv[1])
data = data.pivot("source","target", "attention")
# data.sortlevel(level=0, ascending=True, inplace=True)

ax = sns.heatmap(data, linewidths=0.4)
# ax = sns.heatmap(data, linewidths=0.4, cmap="YlOrRd")


# modify tick labels
# - no position info
# - no byte-pair encoding markers (replace with hyphen)
# - replace COMMA with ,
# - remove CC_ prefix (extended source context)

xlabels = [item.get_text() for item in ax.get_xticklabels()]
ylabels = [item.get_text() for item in ax.get_yticklabels()]

xlabels = [re.sub('^[0-9]*\_','',item) for item in xlabels]
ylabels = [re.sub('^[0-9]*\_','',item) for item in ylabels]

xlabels = [item.replace('COMMA',',') for item in xlabels]
ylabels = [item.replace('COMMA',',') for item in ylabels]

xlabels = [item.replace('@@','-') for item in xlabels]
ylabels = [item.replace('@@','-') for item in ylabels]
ylabels = [re.sub('^CC\_(.*)$','\\1 (context)',item) for item in ylabels]

ax.set_xticklabels(xlabels)
ax.set_yticklabels(ylabels)

sns.plt.yticks(rotation=0)
sns.plt.xticks(rotation=30)

sns.plt.tight_layout()
ax.figure.savefig("attention.pdf")

sns.plt.show()
