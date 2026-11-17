"""Keep Unicode scalar values; render only invalid surrogate code units visibly."""
def repair_surrogates(text):
    if not isinstance(text,str):raise TypeError('Model message content must be a string')
    out=[];repairs=[];i=0
    while i<len(text):
        value=ord(text[i])
        if 0xd800<=value<=0xdbff and i+1<len(text) and 0xdc00<=ord(text[i+1])<=0xdfff:
            out.append(chr(0x10000+((value-0xd800)<<10)+(ord(text[i+1])-0xdc00)))
            repairs.append(dict(offset=i,kind='surrogate_pair'));i+=2;continue
        if 0xd800<=value<=0xdfff:
            out.append('\\u%04x'%value);repairs.append(dict(offset=i,kind='unpaired_surrogate',codepoint=hex(value)))
        else:out.append(text[i])
        i+=1
    return ''.join(out),repairs
