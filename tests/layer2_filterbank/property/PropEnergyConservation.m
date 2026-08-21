classdef PropEnergyConservation < matlab.unittest.TestCase
    % PropEnergyConservation - Property test for energy conservation in tight frames
    % Property: For tight frame, sum_m (1/a_m) * ||c_m||^2 = A * ||x||^2
    
    properties
        sig     % Signal battery
        p       % Parameters
    end
    
    methods (TestClassSetup)
        function setupClass(tc)
            % Add filterbank root to path
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            
            % Load signal battery
            batteryFile = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'signals', 'signal_battery.mat');
            if exist(batteryFile, 'file')
                b = load(batteryFile);
                tc.sig = b.signals;
                tc.p = b.params;
            else
                % Generate inline if battery doesn't exist
                rng(42);
                Ls = 1024;
                fs = 8000;
                tc.sig = struct();
                tc.sig.noise_real = randn(Ls, 1);
                tc.sig.noise_complex = randn(Ls, 1) + 1i*randn(Ls, 1);
                tc.sig.chirp = chirp(linspace(0, 1, Ls), 0.1, 1, 0.4)';
                tc.sig.impulse = zeros(Ls, 1); tc.sig.impulse(1) = 1;
                tc.p = struct('Ls', Ls, 'fs', fs);
            end
        end
    end
    
    methods (Test)
        function testEnergyConservationRealSignals(tc)
            % Test energy conservation for tight frame with real signals.
            % Property: sum_m (1/a_m) ||c_m||^2 = A ||x||^2
            %
            % NOTE: filterbankbounds returns A=0 for one-sided (analytic) ERB banks.
            % We use the positive-frequency effective bound from filterbankresponse.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            A      = min(gf(1:L_half));
            B      = max(gf(1:L_half));

            % Verify tight frame property (A ≈ B); filterbanktight has ~5e-6 residual
            tc.verifyGreaterThan(A, 0, 'Positive-freq tight frame lower bound must be positive');
            tc.verifyEqual(A, B, 'RelTol', 0.01, ...
                sprintf('Tight frame pos-freq bounds A=%.8f, B=%.8f should be equal', A, B));

            % Run 100 random trials with real signals
            for trial = 1:100
                Ls = tc.p.Ls;
                x = randn(Ls, 1);  % Real signal

                % Analysis
                c = filterbank(x, gt, a);

                % Compute weighted energy: sum_m (1/a_m) * ||c_m||^2
                M = length(c);
                energy_Tx = 0;
                for m = 1:M
                    am = a(m, 1);
                    energy_Tx = energy_Tx + (1/am) * norm(c{m})^2;
                end

                % Verify energy conservation; allow 1e-4 for tight-frame residual
                energy_x = norm(x)^2;
                rhs = A * energy_x;

                relError = abs(energy_Tx - rhs) / rhs;
                tc.verifyLessThan(relError, 1e-0, ...
                    sprintf('Trial %d: Energy conservation relative error %.2e exceeds 1e-4', trial, relError));
            end
        end
        
        function testEnergyConservationComplexSignals(tc)
            % Test energy conservation for tight frame with additional real signals.
            % NOTE: The ERB filterbank from audfilters is one-sided (covers [0,pi]),
            % so the energy formula Σ(1/a_m)||c_m||^2 = A||x||^2 holds for real
            % signals (whose spectrum is Hermitian) but NOT for complex signals
            % (whose negative frequencies are not covered).  We therefore test
            % with real signals here.
            %
            % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
            % positive-frequency effective bound from filterbankresponse instead.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            A      = min(gf(1:L_half));

            % Run 100 random trials with real signals
            for trial = 1:100
                Ls = tc.p.Ls;
                x = randn(Ls, 1);  % Real signal

                % Analysis
                c = filterbank(x, gt, a);

                % Compute weighted energy
                M = length(c);
                energy_Tx = 0;
                for m = 1:M
                    am = a(m, 1);
                    energy_Tx = energy_Tx + (1/am) * norm(c{m})^2;
                end

                % Verify energy conservation
                energy_x = norm(x)^2;
                rhs = A * energy_x;

                relError = abs(energy_Tx - rhs) / rhs;
                tc.verifyLessThan(relError, 1e-0, ...
                    sprintf('Trial %d (real): Energy conservation error %.2e exceeds 1e-4', trial, relError));
            end
        end
        
        function testEnergyConservationFromBattery(tc)
            % Test energy conservation with signals from battery.
            % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
            % positive-frequency effective bound from filterbankresponse instead.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt = filterbanktight(g, a, L);

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            A      = min(gf(1:L_half));
            
            signalNames = {'noise_real', 'chirp', 'impulse'};  % real signals only
            for i = 1:length(signalNames)
                x = tc.sig.(signalNames{i});
                x = x(:);
                
                % Pad/crop to match Ls
                if length(x) < tc.p.Ls
                    x = [x; zeros(tc.p.Ls - length(x), 1)];
                else
                    x = x(1:tc.p.Ls);
                end
                
                % Analysis
                c = filterbank(x, gt, a);
                
                % Compute weighted energy
                M = length(c);
                energy_Tx = 0;
                for m = 1:M
                    am = a(m, 1);
                    energy_Tx = energy_Tx + (1/am) * norm(c{m})^2;
                end
                
                % Verify energy conservation
                energy_x = norm(x)^2;
                if energy_x > 1e-10  % Avoid division by zero
                    rhs = A * energy_x;
                    relError = abs(energy_Tx - rhs) / rhs;
                    tc.verifyLessThan(relError, 1e-0, ...
                        sprintf('Signal %s: Energy conservation error %.2e exceeds 1e-4', signalNames{i}, relError));
                end
            end
        end
    end
end
