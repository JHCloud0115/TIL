# Inflearn_데이터 분석을 위한 SQL 실전편 

## Understanding Search Functionality, 검색 기능 분석

검색 기능 효욕 살펴보고 검색 기능에 개선할 점 있는지 확인 
  
**분석 내용**  
1. 검색 기능 사용율 분석
2. 세션 당 검색 기능 사용 횟수 분석
3. 검색 결과 클릭 횟수 분석
4. 검색 결과 순서 배치에 대한 분석 및 재사용율 분석  
  
  
**결론**  
* 자동완성 기능 전체 세션(10분 이상 사용하지 않을 경우 다른 경우로 보고 이를 한 세션이라 함)의 22 - 26% 사용하고,
  full search은 8%정도 사용되고 있는 서비스의 핵심 기능
* full search 사용한 대부분의 세션에서 검색을 2회 이상 반복적 사용
* full search를 한 세션 중 검색 결과 단 한 건도 클릭하지 않은 세션 54.28%
* full search 이후, 검색 결과 페이지에서 유저들이 클릭하는 컨텐츠의 순서 고르게 분포

## #1

```sql
SELECT user_id,
       event_type,
       event_name,
       occurred_at,
       -- partioin by user_id user_id별로 이벤트 발생 순서 시점에 따라 확인 
       LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at),  
       LEAD(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at),
       occurred_at - LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) AS last_event,
       LEAD(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) - occurred_at AS next_event,
       ROW_NUMBER() OVER () AS id
FROM tutorial.yammer_events e
WHERE e.event_type = 'engagement'
AND user_id =20 
ORDER BY user_id,occurred_at;
```

![1.png](attachment:image.png)

--> 2번째 행을 확인해보면 lag는 occured_at의 첫번째 행을 가져와서 이전의 기록을 보여주고  
lead는 그 다음 발생한 occured_at의 시간을 가져와서 보여주는 함수  
last_event는 lead행에서 lag행을 뺀 값  
next_event는 그 다음 행동을 하기까지의 시간을 구한 값


* lag / lead 함수 사용하여 비교하거나 차이를 구할 수 있음

## #2

```sql

SELECT bounds.*,
                        -- 새로운 세션 정의 
              		    CASE WHEN last_event >= INTERVAL '10 MINUTE' THEN id
              		         WHEN last_event IS NULL THEN id
              		         ELSE LAG(id,1) OVER (PARTITION BY user_id ORDER BY occurred_at) END AS session
                FROM (
                     SELECT user_id,
                            event_type,
                            event_name,
                            occurred_at,
                            occurred_at - LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) AS last_event,
                            LEAD(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) - occurred_at AS next_event,
                            ROW_NUMBER() OVER () AS id
                       FROM tutorial.yammer_events e
                      WHERE e.event_type = 'engagement'
                      ORDER BY user_id,occurred_at
                     ) bounds
                -- case에서 나타낸 조건절 
               WHERE last_event >= INTERVAL '10 MINUTE' 
                  OR next_event >= INTERVAL '10 MINUTE'
               	 OR last_event IS NULL
              	 	 OR next_event IS NULL 
```

![2.png](attachment:image.png)

## #3

```sql       
    SELECT user_id,
              session,
            -- 위에서 찾은 내용으로 최대 최소 찾도록 설정 
              MIN(occurred_at) AS session_start, 
              MAX(occurred_at) AS session_end
         FROM (
              SELECT bounds.*,
              		    CASE WHEN last_event >= INTERVAL '10 MINUTE' THEN id
              		         WHEN last_event IS NULL THEN id
              		         ELSE LAG(id,1) OVER (PARTITION BY user_id ORDER BY occurred_at) END AS session
                FROM (
                     SELECT user_id,
                            event_type,
                            event_name,
                            occurred_at,
                            occurred_at - LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) AS last_event,
                            LEAD(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) - occurred_at AS next_event,
                            ROW_NUMBER() OVER () AS id
                       FROM tutorial.yammer_events e
                      WHERE e.event_type = 'engagement'
                      AND user_id =20
                      ORDER BY user_id,occurred_at
                     ) bounds
               WHERE last_event >= INTERVAL '10 MINUTE'
                  OR next_event >= INTERVAL '10 MINUTE'
               	 OR last_event IS NULL
              	 	 OR next_event IS NULL   
              ) final
        GROUP BY 1,2;
```

                                                    ↓       session_start은 occured_at의 최소값   session_end는 occured_at의 최대값

![3.png](attachment:image.png)

## #4

```sql
SELECT DATE_TRUNC('week',z.session_start) AS week, 
       COUNT(CASE WHEN z.autocompletes > 0 THEN z.session ELSE NULL END)/COUNT(*)::FLOAT AS with_autocompletes,
       COUNT(CASE WHEN z.runs > 0 THEN z.session ELSE NULL END)/COUNT(*)::FLOAT AS with_runs
  FROM (
SELECT x.session_start,
       x.session,
       x.user_id,
      -- 각각의 이벤트에 대해서 집계함수 갯수 확인
       COUNT(CASE WHEN x.event_name = 'search_autocomplete' THEN x.user_id ELSE NULL END) AS autocompletes,
       COUNT(CASE WHEN x.event_name = 'search_run' THEN x.user_id ELSE NULL END) AS runs,
       COUNT(CASE WHEN x.event_name LIKE 'search_click_%' THEN x.user_id ELSE NULL END) AS clicks
  FROM (
SELECT e.*,
       session.session,
       session.session_start
  FROM tutorial.yammer_events e
  LEFT JOIN (
       SELECT user_id,
              session,
              MIN(occurred_at) AS session_start,
              MAX(occurred_at) AS session_end
         FROM (
              SELECT bounds.*,
              		    CASE WHEN last_event >= INTERVAL '10 MINUTE' THEN id
              		         WHEN last_event IS NULL THEN id
              		         ELSE LAG(id,1) OVER (PARTITION BY user_id ORDER BY occurred_at) END AS session
                FROM (
                     SELECT user_id,
                            event_type,
                            event_name,
                            occurred_at,
                            occurred_at - LAG(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) AS last_event,
                            LEAD(occurred_at,1) OVER (PARTITION BY user_id ORDER BY occurred_at) - occurred_at AS next_event,
                            ROW_NUMBER() OVER () AS id
                       FROM tutorial.yammer_events e
                      WHERE e.event_type = 'engagement'
                      ORDER BY user_id,occurred_at
                     ) bounds
               WHERE last_event >= INTERVAL '10 MINUTE'
                  OR next_event >= INTERVAL '10 MINUTE'
               	 OR last_event IS NULL
              	 	 OR next_event IS NULL   
              ) final
        GROUP BY 1,2
       ) session
      --user_id가 같을때 occured_at이 세션 시작과 끝 사이에 있어야 함 
    ON e.user_id = session.user_id
   AND e.occurred_at >= session.session_start
   AND e.occurred_at <= session.session_end
 WHERE e.event_type = 'engagement'
       ) x
 GROUP BY 1,2,3
       ) z
 GROUP BY 1
 ORDER BY 1
```

위의 쿼리를 통해 아래와 같은 차트를 얻을 수 있음

![4.png](attachment:image.png)
