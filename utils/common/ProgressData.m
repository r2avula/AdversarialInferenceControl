classdef ProgressData < handle
    properties
        percentDone
        timeRemainingString
        reverseStr
        proc_start
        last_time_updated
        isActive
        Queue
    end
    
    methods
        function [this, Queue] = ProgressData(initialPrintString,startParPool)
            if(nargin==1)
                startParPool = false;
            end
            percentDone = 0;
            timeRemaining = nan;
            if(startParPool)
                gcp;
            end
            fprintf(initialPrintString);
            reverseStr = 'Percent done = ';
            msg = sprintf('%3.2f', percentDone);
            timeRemainingString = sprintf(';\t Est. time left = %ds',timeRemaining);
            msg = [msg,timeRemainingString];
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
            proc_start = tic;
            last_time_updated = -inf;

            this.percentDone = percentDone;
            this.timeRemainingString = timeRemainingString;
            this.reverseStr = reverseStr;
            this.proc_start = proc_start;
            this.last_time_updated = last_time_updated;
            this.isActive = true;

            Queue = parallel.pool.DataQueue;
            afterEach(Queue, @(data)updateProgress(this,data));
            this.Queue = Queue;
        end

        function [proc_time,percentDone] = terminate(this, endString)
            if(nargin==1)
                endString = '';
            end

            percentDone = 100;
            msg = sprintf('%3.2f', percentDone);
            proc_time = toc(this.proc_start);
            if(proc_time>86400)
                msg1 = sprintf(';\t Time taken = %3.1f days.\t%s\n',proc_time/86400,endString);
            elseif(proc_time>3600)
                msg1 = sprintf(';\t Time taken = %3.1f hours.\t%s\n',proc_time/3600,endString);
            elseif(proc_time>60)
                msg1 = sprintf(';\t Time taken = %3.1f minutes.\t%s\n',proc_time/60,endString);
            else
                msg1 = sprintf(';\t Time taken = %3.1f seconds.  \t%s\n',proc_time,endString);
            end
            msg = [msg,msg1];
            fprintf([this.reverseStr, msg]);

            this.isActive = false;
        end

        function  updateProgress(this,incPercent)
            if(this.isActive)
                this.percentDone = this.percentDone + incPercent;
                msg = sprintf('%3.2f', this.percentDone);
                proc_time = toc(this.proc_start);
                if(proc_time-this.last_time_updated >1)
                    timeRemaining = round((proc_time*(100-this.percentDone)/this.percentDone));
                    if(timeRemaining>86400)
                        this.timeRemainingString = sprintf('; Est. time left = %3.1f days   ',timeRemaining/86400);
                    elseif(timeRemaining>3600)
                        this.timeRemainingString = sprintf(';\t Est. time left = %3.1f hours ',timeRemaining/3600);
                    elseif(timeRemaining>60)
                        this.timeRemainingString = sprintf(';\t Est. time left = %d minutes',round(timeRemaining/60));
                    else
                        this.timeRemainingString = sprintf(';\t Est. time left = %d seconds',timeRemaining);
                    end
                    this.last_time_updated = proc_time;
                end
                msg = [msg,this.timeRemainingString];
                fprintf([this.reverseStr, msg]);
                this.reverseStr = repmat(sprintf('\b'), 1, length(msg));

                this.percentDone = this.percentDone;
                this.timeRemainingString = this.timeRemainingString;
                this.reverseStr = this.reverseStr;
                this.last_time_updated = this.last_time_updated;
            end
        end
    end
end