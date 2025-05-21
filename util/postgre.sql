--6. To obtain the full meeting transcript in the correct order, should I order by c.transcript_id, context_id, speaker_number or by c.transcript_id, 
--speaker_number, context_id?  I use this script to get the full meeting transcript, please check it is correct not: 
SELECT 
    c.transcript_id,  -- Select needed CEO data columns
    c.context_id,
    s.context,
    c.speaker_number,
    c.is_qa,
    c.is_ceo,
    s.speaker_id,
    s.speaker_type,
    s.speaker_text
FROM 
    llms.ceo_nor_data c
INNER JOIN
    fs_core.speaker_data s 
        ON c.transcript_id = s.transcript_id 
        AND c.version_id = s.version_id 
        AND c.transcript_type = s.transcript_type 
        AND c.processed_fs = s.processed_fs 
        AND c.context_id = s.context_id 
        AND c.speaker_number = s.speaker_number
-- Add any filtering conditions here
WHERE c.transcript_id = 1438448
order by c.transcript_id, context_id, speaker_number