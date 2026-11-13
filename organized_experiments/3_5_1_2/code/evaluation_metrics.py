"""Task summaries with explicit invalid predictions and LP class coverage."""
def summarize(rows):
    keys=sorted({key for row in rows for key in row['metrics']})
    metrics={key:sum(row['metrics'].get(key,0) for row in rows)/max(1,len(rows)) for key in keys}
    result=dict(count=len(rows),task_accuracy=metrics.get('prediction_correct',0),**metrics)
    lp=[row for row in rows if row.get('ground_truth_polarity') is not None]
    confusion={name:{value:0 for value in ['positive','negative','invalid']} for name in ['positive','negative']}
    for row in lp:
        gold='positive' if row['ground_truth_polarity'] else 'negative'
        predicted=row.get('prediction_polarity')
        value='positive' if predicted is True else 'negative' if predicted is False else 'invalid'
        confusion[gold][value]+=1
    pos=sum(confusion['positive'].values());neg=sum(confusion['negative'].values())
    tp=confusion['positive']['positive'];tn=confusion['negative']['negative']
    fp=confusion['negative']['positive'];fn=pos-tp
    negative_false_positive=confusion['positive']['negative'];negative_false_negative=neg-tn
    positive_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    negative_f1=2*tn/(2*tn+negative_false_positive+negative_false_negative) if 2*tn+negative_false_positive+negative_false_negative else 0.0
    result['lp']=dict(available=bool(lp),count=len(lp),positive_count=pos,negative_count=neg,
        both_gold_classes_present=bool(pos and neg),confusion=confusion,
        invalid_predictions=confusion['positive']['invalid']+confusion['negative']['invalid'],
        positive_recall=tp/pos if pos else None,negative_recall=tn/neg if neg else None,
        balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,
        macro_f1=(positive_f1+negative_f1)/2 if pos and neg else None)
    return result
