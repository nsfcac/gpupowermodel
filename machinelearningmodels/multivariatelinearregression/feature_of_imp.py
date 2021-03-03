# import pandas as pd
# import matplotlib.pyplot as plt
# plt.figure()
# plt.style.use('classic')
# min = [9, 4, 4, 3,3,3,3]
# men = [20,18,17,10,10,10,10]
# max = [42, 47, 45, 21,22,21,18]
# index = ['2-Features', '3-Features', '4-Features','5-Features', '6-Features', '7-Features','8-Features']
# df = pd.DataFrame({'Min % Error': min,'Mean % Error': men,'Max % Error':max}, index=index)
# ax = df.plot.bar(color = 'rgbkymc',rot=30)
# ax.set_xlabel("No. of Features",weight='bold',fontsize=14)
# ax.set_ylabel('% Error',weight='bold',fontsize=14)
# plt.grid(True)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.tight_layout()
# plt.savefig('C:/rf/prediction-error-with-non-standard-features.png')
# plt.show()

# plt.figure()
# plt.style.use('classic')

# min = [10, 12, 11, 12,11,12,20]
# men = [36,38,38,39,38,38,84]
# max = [119, 118, 118, 117,116,116,289]

# index = ['2-Features', '3-Features', '4-Features','5-Features', '6-Features', '7-Features','8-Features']
# df = pd.DataFrame({'Min % Error': min,
#                    'Mean % Error': men,
#                     'Max % Error':max}, index=index)
# ax = df.plot.bar(color = 'rgbkymc',rot=30)
# # plt.ylim([0, 130])
# plt.legend(loc=0)
# ax.set_xlabel("No. of Features",weight='bold',fontsize=14)
# ax.set_ylabel('% Error',weight='bold',fontsize=14)
# plt.grid(True)
# plt.tick_params(axis='both', which='major', labelsize=14)
# plt.tight_layout()
# plt.savefig('C:/rf/prediction-error-with-standard-features.png')

# plt.show()

